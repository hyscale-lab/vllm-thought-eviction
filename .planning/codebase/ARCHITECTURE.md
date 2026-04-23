# Architecture

**Analysis Date:** 2026-04-07

## Pattern Overview

**Overall:** Layered middleware extension on top of vLLM's v1 async inference engine

**Key Characteristics:**
- Thought eviction runs as async middleware wrapping the streaming token generator — the engine core is never blocked
- One `EvictionOrchestrator` instance is created per request; all mutable state is strictly isolated per instance (no shared state between requests)
- L2 norms are computed in the GPU model runner (worker process) and propagated back to the API layer via IPC through the `EngineCoreOutput` data structure
- Eviction commands travel from the API layer back down through the engine core to the scheduler via `update_request_mask`, completing a round-trip loop
- The `L2NormCache` is a global singleton in the worker process, keyed by request ID

## Layers

**API / Entrypoint Layer:**
- Purpose: Accepts chat completion requests, instantiates `EvictionOrchestrator`, wraps the result stream, and emits eviction statistics in the final SSE chunk
- Location: `vllm/entrypoints/openai/chat_completion/serving.py`
- Contains: `OpenAIServingChat`, stream wrapping logic, eviction payload assembly
- Depends on: `EvictionOrchestrator`, `EngineClient`, `EvictionParams`
- Used by: HTTP clients via FastAPI

**Eviction Orchestrator Layer:**
- Purpose: Per-request async middleware that accumulates L2 norms and text from stream tokens, segments reasoning into thoughts, selects ranges for eviction, and issues eviction commands to the engine
- Location: `vllm/thought_eviction/orchestrator.py`
- Contains: `EvictionOrchestrator` class
- Depends on: `ThoughtSegmenter`, strategy classes, `block_utils`, `L2NormCache` (for cleanup), `engine_client.update_request_mask`
- Used by: `vllm/entrypoints/openai/chat_completion/serving.py`

**Thought Segmentation Layer:**
- Purpose: Parses `<think>...</think>` reasoning content into discrete `ThoughtSegment` objects using 14 linguistic boundary phrases; computes reasoning-relative token positions via tokenizer `offset_mapping`
- Location: `vllm/thought_eviction/segmenter.py`
- Contains: `ThoughtSegmenter`, `ThoughtSegment` dataclass
- Depends on: tokenizer (passed in; never loaded from disk)
- Used by: `EvictionOrchestrator`

**Eviction Strategy Layer:**
- Purpose: Stateless (or stably-scored) algorithms that receive thoughts or raw L2 norm arrays and return reasoning-relative `(start, end)` token ranges to evict
- Location: `vllm/thought_eviction/strategies.py`
- Contains: `GlobalStrategy`, `ThoughtMinStrategy`, `ThoughtAvgStrategy`, `RandomStrategy`
- Depends on: `ThoughtSegment`, `numpy`
- Used by: `EvictionOrchestrator`

**Block Utilities Layer:**
- Purpose: Pure functions for preparing raw token ranges for physical KV cache block eviction: merge, block-align, retention-window trim
- Location: `vllm/thought_eviction/block_utils.py`
- Contains: `merge_overlapping_ranges`, `align_ranges_to_blocks`, `apply_retention_window`
- Depends on: nothing external
- Used by: `EvictionOrchestrator`

**Engine Core / Scheduler Layer:**
- Purpose: Accepts `update_request_mask` calls from the API layer, stores eviction ranges per request, and passes them to the model runner as `evictable_token_ranges_map` inside `SchedulerOutput`
- Location: `vllm/v1/core/sched/scheduler.py`, `vllm/v1/core/sched/output.py`
- Contains: `update_request_mask`, `_process_evictions`, `SchedulerOutput.evictable_token_ranges_map`
- Depends on: KV cache manager
- Used by: `vllm/v1/engine/core.py`, `vllm/v1/worker/gpu_model_runner.py`

**GPU Model Runner / L2 Norm Computation Layer:**
- Purpose: Reads `scheduler_output.evictable_token_ranges_map` to invalidate KV cache blocks; after each forward pass calls `_compute_l2_norms` to compute mean L2 norms from the key cache across selected layers and stores them in `L2NormCache`
- Location: `vllm/v1/worker/gpu_model_runner.py`
- Contains: block invalidation logic (`evicted_ranges`), `_compute_l2_norms`, L2 norm cache integration
- Depends on: `L2NormCache` (singleton), `flash_attn` backend metadata
- Used by: Engine forward loop

**L2 Norm Cache Layer:**
- Purpose: Thread-safe global singleton (worker process) that stores per-request L2 norm buffers; exposes differential retrieval (`get_norms(start_index)`) and per-request layer filtering
- Location: `vllm/v1/attention/l2_norm_cache.py`
- Contains: `L2NormCache`, `RequestL2NormData`, `get_l2_norm_cache()`
- Depends on: `torch`, `threading`
- Used by: `gpu_model_runner.py`, `scheduler.py` (retrieval), `orchestrator.py` (cleanup)

**Engine Output Pipeline:**
- Purpose: Reads differential L2 norms from `L2NormCache` per request and attaches them as `new_l2_norms` to `EngineCoreOutput`, which is then propagated to `RequestOutput` through the output processor
- Location: `vllm/v1/core/sched/scheduler.py` (norm retrieval), `vllm/v1/engine/output_processor.py` (propagation), `vllm/v1/engine/__init__.py` (`EngineCoreOutput`)
- Contains: `EngineCoreOutput.new_l2_norms`, output processor norm forwarding
- Depends on: `L2NormCache`
- Used by: `EvictionOrchestrator._accumulate()`

## Data Flow

**Forward Pass — L2 Norm Computation:**

1. HTTP request arrives at `vllm/entrypoints/openai/chat_completion/serving.py`
2. `eviction_params` is extracted; `sampling_params.enable_l2_norms = True` and `sampling_params.l2_norm_layers` are set on the `SamplingParams`
3. `SamplingParams` travels via IPC to the engine core worker process
4. During each forward pass, `gpu_model_runner.py` detects `enable_l2_norms` and calls `_compute_l2_norms()` after the forward pass
5. `_compute_l2_norms` iterates over the KV key cache, computes per-layer L2 norms, averages them across selected layers, and calls `L2NormCache.update_norms_batch()`
6. `L2NormCache` stores norms in a pre-allocated `torch.Tensor` buffer per request

**Return Path — Norm Delivery:**

1. `scheduler.py` retrieves differential norms from `L2NormCache.get_norms(start_index)` and attaches them to `EngineCoreOutput.new_l2_norms`
2. `output_processor.py` propagates `new_l2_norms` into `RequestOutput`
3. `EvictionOrchestrator._accumulate()` extends `self.accumulated_l2_norms` differentially from `res.new_l2_norms`

**Eviction Cycle — Command Path:**

1. `EvictionOrchestrator._maybe_schedule_cycle()` checks the configured trigger (time or token count)
2. `_run_eviction_cycle()` fires as a background `asyncio.Task` (non-blocking to the stream)
3. `ThoughtSegmenter.update()` segments current reasoning content into `ThoughtSegment` objects
4. L2 norms are sliced from `accumulated_l2_norms` using `reasoning_start_token_offset` and assigned to each segment
5. The strategy computes reasoning-relative eviction ranges
6. Ranges are merged, retention-windowed, and block-aligned by `block_utils`
7. Ranges are converted to absolute token positions using `reasoning_start_token_offset`
8. `engine_client.update_request_mask(request_id, absolute_ranges)` is awaited
9. IPC transports the mask update to `scheduler.update_request_mask()` → stored in `request_eviction_data`
10. On the next scheduler tick, `_process_evictions()` places ranges in `SchedulerOutput.evictable_token_ranges_map`
11. `gpu_model_runner.py` invalidates the corresponding KV cache blocks and replaces them with attention-sink copies

**State Management:**
- All per-request state lives in `EvictionOrchestrator` (API process) and `RequestL2NormData` (worker process)
- `EvictionOrchestrator` is discarded when the stream ends; `L2NormCache.remove_request()` is called in the `finally` block of `wrap_stream`
- `RandomStrategy` maintains stable per-thought scores via `_thought_random_scores` dict keyed by `start_char_pos`

## Key Abstractions

**ThoughtSegment:**
- Purpose: A single reasoning unit bounded by linguistic separator phrases; carries char positions, reasoning-relative token positions, L2 norms, and eviction status
- Examples: `vllm/thought_eviction/segmenter.py`
- Pattern: `@dataclass` with mutable `l2_norms`, `min_l2_norm`, `avg_l2_norm`, `evicted` fields

**EvictionParams:**
- Purpose: API-layer configuration object that controls all aspects of eviction for a single request: strategy selection, keep ratio, trigger mode, retention window, layer selection
- Examples: `vllm/entrypoints/openai/chat_completion/protocol.py`
- Pattern: Pydantic `OpenAIBaseModel`; embedded as optional field in `ChatCompletionRequest`

**L2NormCache (singleton):**
- Purpose: Cross-thread data store bridging the GPU forward pass and the output pipeline; holds pre-allocated per-token float32 buffers per request
- Examples: `vllm/v1/attention/l2_norm_cache.py`
- Pattern: Thread-safe singleton via double-checked locking; accessed via `get_l2_norm_cache()`

**EvictionOrchestrator:**
- Purpose: Per-request stateful middleware; owns the full eviction lifecycle from text accumulation through strategy selection through engine command dispatch
- Examples: `vllm/thought_eviction/orchestrator.py`
- Pattern: Instantiated in `serving.py`; wraps the async result generator via `wrap_stream()`

**Strategy Protocol:**
- Purpose: Each strategy implements a single method `compute_evictable_ranges(...)` returning reasoning-relative `list[tuple[int, int]]`
- Examples: `vllm/thought_eviction/strategies.py` — `GlobalStrategy`, `ThoughtMinStrategy`, `ThoughtAvgStrategy`, `RandomStrategy`
- Pattern: Duck-typed (no formal ABC); strategies are instantiated via `_build_strategy(params)` factory

## Entry Points

**HTTP API (OpenAI-compatible):**
- Location: `vllm/entrypoints/openai/chat_completion/serving.py` — `OpenAIServingChat.create_chat_completion()`
- Triggers: POST `/v1/chat/completions` with `eviction_params` field and `stream: true`
- Responsibilities: Validates eviction constraints (stream required), creates `EvictionOrchestrator`, wraps result generator, returns SSE stream with final eviction stats chunk

**EvictionOrchestrator.wrap_stream():**
- Location: `vllm/thought_eviction/orchestrator.py`
- Triggers: Called once per streaming request with `eviction_params`
- Responsibilities: Intercepts each `RequestOutput`, accumulates L2 norms and text, schedules eviction cycles as background tasks, cleans up on completion

**Scheduler.update_request_mask():**
- Location: `vllm/v1/core/sched/scheduler.py`
- Triggers: Called via IPC from `engine_client.update_request_mask_async()` during an active eviction cycle
- Responsibilities: Stores eviction ranges; they are flushed into `SchedulerOutput` on the next scheduler tick

**GPUModelRunner._compute_l2_norms():**
- Location: `vllm/v1/worker/gpu_model_runner.py`
- Triggers: Called after each forward pass when at least one request in the batch has `enable_l2_norms=True`
- Responsibilities: Iterates KV key cache tensors across selected layers, computes mean L2 norm per token, writes to `L2NormCache`

## Error Handling

**Strategy:** Non-fatal; eviction cycle errors are caught and logged without crashing the stream generator

**Patterns:**
- `_run_eviction_cycle()` wraps the entire body in `try/except Exception` — errors are logged via `logger.error(..., exc_info=True)` and the cycle is silently skipped
- `asyncio.CancelledError` is re-raised so task cancellation (client disconnect) works correctly
- Guard conditions at the top of `_run_eviction_cycle()` return early without error when preconditions are not met (insufficient norms, delay intervals, no reasoning content, offset unknown)
- `L2NormCache.update_norms_batch()` catches all exceptions and logs them as info to avoid crashing the forward pass

## Cross-Cutting Concerns

**Logging:** `vllm.logger.init_logger(__name__)` in every module; eviction cycle completions are logged at INFO with cycle count, request ID, range count, and token count
**Validation:** `EvictionParams` uses Pydantic field validators (`ge`, `le`, `gt`); non-streaming requests with `eviction_params` are rejected at the API layer with an error response
**Authentication:** Inherited from base vLLM; no eviction-specific auth
**IPC Boundary:** `SamplingParams.enable_l2_norms` and `SamplingParams.l2_norm_layers` cross the API→worker IPC boundary; `update_request_mask` and `get_request_l2_norms` cross the worker→API IPC boundary via `call_utility_async`

---

*Architecture analysis: 2026-04-07*
