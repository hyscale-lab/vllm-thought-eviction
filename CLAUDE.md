<!-- GSD:project-start source:PROJECT.md -->
## Project

**vLLM Thought Eviction — Upgrade to v0.19.0**

A fork of vLLM (hyscale-lab/vllm-thought-eviction) that adds thought eviction with L2 norms to the vLLM inference engine. Currently based on vLLM v0.14.0rc2. The goal of this project is to upgrade the fork to vLLM v0.19.0 (commit 2a69949) while preserving all custom thought eviction functionality.

**Core Value:** The upgrade must preserve working thought eviction — merging upstream v0.19.0 changes without breaking the eviction orchestrator, L2 norm pipeline, or strategy system.

### Constraints

- **Compatibility**: Eviction code hooks into vLLM internals (scheduler, sampler, worker IPC) — these are the most likely conflict zones
- **Branch**: All work happens on `upgrade_vllm` branch before PR to main
- **Verification**: Tests must pass AND server must start with eviction requests working
<!-- GSD:project-end -->

<!-- GSD:stack-start source:codebase/STACK.md -->
## Technology Stack

## Languages
- Python 3.12 (runtime on this machine; supported range 3.10–3.13) - All vLLM engine, serving, and thought eviction logic
- C++17 - CUDA/HIP extension kernels in `csrc/`
- CUDA C (.cu) - GPU attention, cache, norm, and sampling kernels in `csrc/`
- CMake - Build system for C++/CUDA extensions (`CMakeLists.txt`, `cmake/`)
- Protobuf / gRPC IDL - Engine IPC definition at `vllm/grpc/vllm_engine.proto`
- Jinja2 - Chat template rendering (`examples/template_*.jinja`)
## Runtime
- CPython 3.10–3.13 (3.12 installed on this machine)
- Linux (primary) or macOS (CPU-only fallback); WSL supported
- pip / uv (uv config present in `pyproject.toml` `[tool.uv]`)
- No lockfile committed; `requirements/*.txt` pin versions
## Frameworks
- FastAPI >= 0.115.0 - OpenAI-compatible HTTP API server (`vllm/entrypoints/openai/`)
- Uvicorn - ASGI server, launched in `vllm/entrypoints/launcher.py`
- PyTorch 2.9.1 - Tensor ops, CUDA integration, model execution (all of `vllm/model_executor/`, `vllm/v1/`)
- FlashInfer-Python 0.5.3 - Paged attention kernels (`vllm/v1/attention/backends/flashinfer.py`)
- Numba 0.61.2 - N-gram speculative decoding (`requirements/cuda.txt`)
- NumPy - L2 norm arrays in thought eviction (`vllm/thought_eviction/strategies.py`, `vllm/thought_eviction/orchestrator.py`, `vllm/v1/attention/l2_norm_cache.py`)
- Triton (bundled with PyTorch + vendored in `vllm/third_party/triton_kernels/`) - Custom GPU kernels for MoE, topk, matmul
- Ray >= 2.48.0 with `ray[cgraph]` - Pipeline parallelism, multi-node worker lifecycle (`vllm/v1/engine/utils.py`, `vllm/ray/`)
- ZeroMQ (pyzmq >= 25.0.0) - Inter-process communication between engine core and API server (`vllm/v1/engine/core.py`, `vllm/v1/engine/core_client.py`)
- msgspec + msgpack - Zero-copy IPC serialization (`vllm/v1/serial_utils.py`, `vllm/v1/engine/coordinator.py`)
- HuggingFace Transformers >= 4.56.0 - Model config, tokenizer base (`vllm/transformers_utils/`)
- HuggingFace tokenizers >= 0.21.1 - Fast incremental detokenization
- sentencepiece - LLaMA tokenizer
- tiktoken >= 0.6.0 - DBRX tokenizer (`vllm/tokenizers/grok2.py`)
- mistral_common >= 1.8.8 - Mistral/Tekken tokenizer and instruct protocol (`vllm/tokenizers/mistral.py`)
- Pydantic >= 2.12.0 - Request/response models throughout entrypoints and thought eviction params (`vllm/entrypoints/openai/chat_completion/protocol.py`)
- grpcio >= 1.76.0 - Engine gRPC server (`vllm/entrypoints/grpc_server.py`)
- grpcio-tools >= 1.76.0 - Proto compilation at build time (`setup.py`)
- grpcio-reflection >= 1.76.0 - gRPC server reflection
- MCP (model context protocol) - `vllm/entrypoints/mcp/`
- xgrammar 0.1.29 - Grammar-constrained decoding
- outlines_core 0.2.11 - Outlines-based structured output
- lm-format-enforcer 0.11.3 - Format-enforced generation
- llguidance >= 1.3.0 - Guidance-based backend (`vllm/v1/structured_output/backend_guidance.py`)
- pytest - Test runner (configured in `pyproject.toml` `[tool.pytest.ini_options]`)
- Ruff - Linting and formatting (configured in `pyproject.toml` `[tool.ruff]`)
- mypy - Static type checking (configured in `pyproject.toml` `[tool.mypy]`)
- pre-commit 4.0.1 - Git hooks (`requirements/lint.txt`, `.pre-commit-config.yaml`)
- typos - Spell checker (configured in `pyproject.toml` `[tool.typos]`)
## Key Dependencies
- `torch==2.9.1` - All GPU compute; pinned exactly (`requirements/cuda.txt`, `pyproject.toml`)
- `flashinfer-python==0.5.3` - Paged attention for CUDA path; must match torch version
- `ray[cgraph]>=2.48.0` - Required for pipeline parallelism
- `pyzmq>=25.0.0` - Engine IPC backbone
- `transformers>=4.56.0,<5` - Model loading and tokenization
- `numpy` - L2 norm array operations in `vllm/thought_eviction/strategies.py` and `vllm/v1/attention/l2_norm_cache.py`
- `torch` (CPU tensors) - `RequestL2NormData` pre-allocates a 120KB CPU `torch.zeros` buffer per request in `vllm/v1/attention/l2_norm_cache.py`
- `prometheus_client>=0.18.0` - Metrics (`vllm/v1/metrics/`)
- `prometheus-fastapi-instrumentator>=7.0.0` - HTTP server metrics
- `pydantic>=2.12.0` - All API models including `EvictionParams` in `vllm/entrypoints/openai/chat_completion/protocol.py`
- `msgspec` - Fast IPC serialization for scheduler ↔ engine boundary
- `fastapi[standard]>=0.115.0` - HTTP serving layer
## Configuration
- `VLLM_TARGET_DEVICE` - Controls build target: `cuda`, `rocm`, `cpu`, `tpu`, `xpu`; auto-detected from torch in `setup.py`
- `VLLM_DISABLE_SCCACHE` - Disable sccache for compilation
- Environment variables loaded from `vllm/envs.py` at build time
- `pyproject.toml` - Project metadata, build backend, ruff/mypy/pytest config
- `CMakeLists.txt` - C++/CUDA extension build (requires cmake >= 3.26.1)
- `cmake/cpu_extension.cmake`, `cmake/utils.cmake` - Platform-specific build helpers
- `requirements/build.txt` - Build-time Python dependencies (cmake, ninja, torch, jinja2, grpcio-tools)
- `setup.py` - Custom build logic; compiles gRPC protos, dispatches to CMake
## Platform Requirements
- Linux or macOS (Linux required for CUDA/ROCm)
- CUDA toolkit (for CUDA target) - `CUDA_HOME` must be set; supported archs 7.0–12.0 depending on nvcc version
- ROCm (for ROCm target) - `ROCM_HOME`; `cmake/hipify.py` converts CUDA sources
- Ninja build tool (recommended)
- NVIDIA GPU (CUDA 12.x or 13.x recommended for sm_90+ / Hopper)
- AMD GPU (ROCm path via `csrc/rocm/`, `requirements/rocm.txt`)
- Google TPU (optional, `requirements/tpu.txt`)
- Intel GPU/XPU (optional, `requirements/xpu.txt`)
- Deployable via Docker (`docker/Dockerfile`, `docker/Dockerfile.rocm`, etc.)
<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->
## Conventions

## Naming Patterns
- `snake_case.py` for all Python modules: `block_utils.py`, `orchestrator.py`, `segmenter.py`, `strategies.py`
- Test files prefixed with `test_`: `test_strategies.py`, `test_orchestrator.py`, `test_block_utils.py`
- Private module-level constants use `_UPPER_SNAKE_CASE` with leading underscore: `_THINK_START_RE`, `_THINK_END_RE`, `_SEPARATOR_OVERLAP`
- `snake_case` for all functions and methods: `compute_evictable_ranges`, `merge_overlapping_ranges`, `align_ranges_to_blocks`
- Private helpers prefixed with `_`: `_indices_to_ranges`, `_accumulate`, `_maybe_schedule_cycle`, `_run_eviction_cycle`, `_build_strategy`
- Private helper functions with `_` prefix at module level when not part of a class's public API: `_indices_to_ranges` in `vllm/thought_eviction/strategies.py`
- `snake_case` throughout: `l2_norms`, `keep_ratio`, `prune_after_tokens`, `eviction_candidates`
- Private instance attributes use leading `_`: `_pending_task`, `_generation_finished`, `_in_think_block`, `_think_start_found`, `_eviction_events`, `_thought_random_scores`
- `PascalCase` for all classes: `EvictionOrchestrator`, `ThoughtSegmenter`, `ThoughtSegment`, `GlobalStrategy`, `ThoughtMinStrategy`, `ThoughtAvgStrategy`, `RandomStrategy`
- Module-level class attributes in `UPPER_SNAKE_CASE`: `TARGET_PHRASES`, `_SEPARATOR_OVERLAP` in `vllm/thought_eviction/segmenter.py`
- Private module-level compiled regexes: `_THINK_START_RE`, `_THINK_END_RE` in `vllm/thought_eviction/orchestrator.py`
## Code Style
- Config in `pyproject.toml` under `[tool.ruff.format]`
- `docstring-code-format = true` — code blocks inside docstrings are also formatted
- Rules enabled: pycodestyle (E), Pyflakes (F), pyupgrade (UP), flake8-bugbear (B), flake8-simplify (SIM), isort (I), flake8-logging-format (G)
- Notable ignores: star imports (F403, F405), lambda assignment (E731), zip without strict (B905)
- `vllm/third_party/**` is excluded from all rules
- `ignore_missing_imports = true`, `check_untyped_defs = true`, `follow_imports = "silent"`
- Runs for Python 3.10, 3.11, 3.12, 3.13 in CI (manual stage); 3.10 locally on pre-commit
## SPDX License Headers
## Import Organization
- `import regex as re` required instead of `import re` (enforced by `enforce-import-regex-instead-of-re` hook)
- Direct `import triton` is forbidden (use `vllm.utils.custom_op` wrappers)
- `pickle`/`cloudpickle` imports are blocked
- Root `vllm/__init__.py` must use lazy imports only (enforced by `check-root-lazy-imports`)
## Docstrings
## Type Annotations
- Use `list[tuple[int, int]]` (not `List[Tuple[int, int]]`) for builtin generics
- Use `Optional[int]` from `typing` for nullable values (not `int | None` style)
- Return type annotations on all public methods: `-> None`, `-> list[tuple[int, int]]`, etc.
- `from typing import Optional` is explicitly imported in files that use it
## Logging
- `logger.info(...)` for successful eviction cycle completion
- `logger.error(..., exc_info=True)` for caught exceptions in eviction cycles
## Error Handling
- Async eviction cycles catch all exceptions to avoid crashing the stream: `except Exception as exc: logger.error(...)`
- `asyncio.CancelledError` is always re-raised: `except asyncio.CancelledError: raise`
- `ValueError` is raised for unknown strategy names in `_build_strategy`
- Early-return guards (not exceptions) used for precondition failures: token threshold, delay intervals, missing state
## Comments
## Function Design
## Module Design
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->
## Architecture

## Pattern Overview
- Thought eviction runs as async middleware wrapping the streaming token generator — the engine core is never blocked
- One `EvictionOrchestrator` instance is created per request; all mutable state is strictly isolated per instance (no shared state between requests)
- L2 norms are computed in the GPU model runner (worker process) and propagated back to the API layer via IPC through the `EngineCoreOutput` data structure
- Eviction commands travel from the API layer back down through the engine core to the scheduler via `update_request_mask`, completing a round-trip loop
- The `L2NormCache` is a global singleton in the worker process, keyed by request ID
## Layers
- Purpose: Accepts chat completion requests, instantiates `EvictionOrchestrator`, wraps the result stream, and emits eviction statistics in the final SSE chunk
- Location: `vllm/entrypoints/openai/chat_completion/serving.py`
- Contains: `OpenAIServingChat`, stream wrapping logic, eviction payload assembly
- Depends on: `EvictionOrchestrator`, `EngineClient`, `EvictionParams`
- Used by: HTTP clients via FastAPI
- Purpose: Per-request async middleware that accumulates L2 norms and text from stream tokens, segments reasoning into thoughts, selects ranges for eviction, and issues eviction commands to the engine
- Location: `vllm/thought_eviction/orchestrator.py`
- Contains: `EvictionOrchestrator` class
- Depends on: `ThoughtSegmenter`, strategy classes, `block_utils`, `L2NormCache` (for cleanup), `engine_client.update_request_mask`
- Used by: `vllm/entrypoints/openai/chat_completion/serving.py`
- Purpose: Parses `<think>...</think>` reasoning content into discrete `ThoughtSegment` objects using 14 linguistic boundary phrases; computes reasoning-relative token positions via tokenizer `offset_mapping`
- Location: `vllm/thought_eviction/segmenter.py`
- Contains: `ThoughtSegmenter`, `ThoughtSegment` dataclass
- Depends on: tokenizer (passed in; never loaded from disk)
- Used by: `EvictionOrchestrator`
- Purpose: Stateless (or stably-scored) algorithms that receive thoughts or raw L2 norm arrays and return reasoning-relative `(start, end)` token ranges to evict
- Location: `vllm/thought_eviction/strategies.py`
- Contains: `GlobalStrategy`, `ThoughtMinStrategy`, `ThoughtAvgStrategy`, `RandomStrategy`
- Depends on: `ThoughtSegment`, `numpy`
- Used by: `EvictionOrchestrator`
- Purpose: Pure functions for preparing raw token ranges for physical KV cache block eviction: merge, block-align, retention-window trim
- Location: `vllm/thought_eviction/block_utils.py`
- Contains: `merge_overlapping_ranges`, `align_ranges_to_blocks`, `apply_retention_window`
- Depends on: nothing external
- Used by: `EvictionOrchestrator`
- Purpose: Accepts `update_request_mask` calls from the API layer, stores eviction ranges per request, and passes them to the model runner as `evictable_token_ranges_map` inside `SchedulerOutput`
- Location: `vllm/v1/core/sched/scheduler.py`, `vllm/v1/core/sched/output.py`
- Contains: `update_request_mask`, `_process_evictions`, `SchedulerOutput.evictable_token_ranges_map`
- Depends on: KV cache manager
- Used by: `vllm/v1/engine/core.py`, `vllm/v1/worker/gpu_model_runner.py`
- Purpose: Reads `scheduler_output.evictable_token_ranges_map` to invalidate KV cache blocks; after each forward pass calls `_compute_l2_norms` to compute mean L2 norms from the key cache across selected layers and stores them in `L2NormCache`
- Location: `vllm/v1/worker/gpu_model_runner.py`
- Contains: block invalidation logic (`evicted_ranges`), `_compute_l2_norms`, L2 norm cache integration
- Depends on: `L2NormCache` (singleton), `flash_attn` backend metadata
- Used by: Engine forward loop
- Purpose: Thread-safe global singleton (worker process) that stores per-request L2 norm buffers; exposes differential retrieval (`get_norms(start_index)`) and per-request layer filtering
- Location: `vllm/v1/attention/l2_norm_cache.py`
- Contains: `L2NormCache`, `RequestL2NormData`, `get_l2_norm_cache()`
- Depends on: `torch`, `threading`
- Used by: `gpu_model_runner.py`, `scheduler.py` (retrieval), `orchestrator.py` (cleanup)
- Purpose: Reads differential L2 norms from `L2NormCache` per request and attaches them as `new_l2_norms` to `EngineCoreOutput`, which is then propagated to `RequestOutput` through the output processor
- Location: `vllm/v1/core/sched/scheduler.py` (norm retrieval), `vllm/v1/engine/output_processor.py` (propagation), `vllm/v1/engine/__init__.py` (`EngineCoreOutput`)
- Contains: `EngineCoreOutput.new_l2_norms`, output processor norm forwarding
- Depends on: `L2NormCache`
- Used by: `EvictionOrchestrator._accumulate()`
## Data Flow
- All per-request state lives in `EvictionOrchestrator` (API process) and `RequestL2NormData` (worker process)
- `EvictionOrchestrator` is discarded when the stream ends; `L2NormCache.remove_request()` is called in the `finally` block of `wrap_stream`
- `RandomStrategy` maintains stable per-thought scores via `_thought_random_scores` dict keyed by `start_char_pos`
## Key Abstractions
- Purpose: A single reasoning unit bounded by linguistic separator phrases; carries char positions, reasoning-relative token positions, L2 norms, and eviction status
- Examples: `vllm/thought_eviction/segmenter.py`
- Pattern: `@dataclass` with mutable `l2_norms`, `min_l2_norm`, `avg_l2_norm`, `evicted` fields
- Purpose: API-layer configuration object that controls all aspects of eviction for a single request: strategy selection, keep ratio, trigger mode, retention window, layer selection
- Examples: `vllm/entrypoints/openai/chat_completion/protocol.py`
- Pattern: Pydantic `OpenAIBaseModel`; embedded as optional field in `ChatCompletionRequest`
- Purpose: Cross-thread data store bridging the GPU forward pass and the output pipeline; holds pre-allocated per-token float32 buffers per request
- Examples: `vllm/v1/attention/l2_norm_cache.py`
- Pattern: Thread-safe singleton via double-checked locking; accessed via `get_l2_norm_cache()`
- Purpose: Per-request stateful middleware; owns the full eviction lifecycle from text accumulation through strategy selection through engine command dispatch
- Examples: `vllm/thought_eviction/orchestrator.py`
- Pattern: Instantiated in `serving.py`; wraps the async result generator via `wrap_stream()`
- Purpose: Each strategy implements a single method `compute_evictable_ranges(...)` returning reasoning-relative `list[tuple[int, int]]`
- Examples: `vllm/thought_eviction/strategies.py` — `GlobalStrategy`, `ThoughtMinStrategy`, `ThoughtAvgStrategy`, `RandomStrategy`
- Pattern: Duck-typed (no formal ABC); strategies are instantiated via `_build_strategy(params)` factory
## Entry Points
- Location: `vllm/entrypoints/openai/chat_completion/serving.py` — `OpenAIServingChat.create_chat_completion()`
- Triggers: POST `/v1/chat/completions` with `eviction_params` field and `stream: true`
- Responsibilities: Validates eviction constraints (stream required), creates `EvictionOrchestrator`, wraps result generator, returns SSE stream with final eviction stats chunk
- Location: `vllm/thought_eviction/orchestrator.py`
- Triggers: Called once per streaming request with `eviction_params`
- Responsibilities: Intercepts each `RequestOutput`, accumulates L2 norms and text, schedules eviction cycles as background tasks, cleans up on completion
- Location: `vllm/v1/core/sched/scheduler.py`
- Triggers: Called via IPC from `engine_client.update_request_mask_async()` during an active eviction cycle
- Responsibilities: Stores eviction ranges; they are flushed into `SchedulerOutput` on the next scheduler tick
- Location: `vllm/v1/worker/gpu_model_runner.py`
- Triggers: Called after each forward pass when at least one request in the batch has `enable_l2_norms=True`
- Responsibilities: Iterates KV key cache tensors across selected layers, computes mean L2 norm per token, writes to `L2NormCache`
## Error Handling
- `_run_eviction_cycle()` wraps the entire body in `try/except Exception` — errors are logged via `logger.error(..., exc_info=True)` and the cycle is silently skipped
- `asyncio.CancelledError` is re-raised so task cancellation (client disconnect) works correctly
- Guard conditions at the top of `_run_eviction_cycle()` return early without error when preconditions are not met (insufficient norms, delay intervals, no reasoning content, offset unknown)
- `L2NormCache.update_norms_batch()` catches all exceptions and logs them as info to avoid crashing the forward pass
## Cross-Cutting Concerns
<!-- GSD:architecture-end -->

<!-- GSD:workflow-start source:GSD defaults -->
## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:
- `/gsd:quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd:debug` for investigation and bug fixing
- `/gsd:execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->



<!-- GSD:profile-start -->
## Developer Profile

> Profile not yet configured. Run `/gsd:profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
