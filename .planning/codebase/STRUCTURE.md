# Codebase Structure

**Analysis Date:** 2026-04-07

## Directory Layout

```
vllm-thought-eviction/
├── vllm/                           # Main Python package
│   ├── thought_eviction/           # Custom eviction module (new in this fork)
│   │   ├── __init__.py             # Package marker
│   │   ├── orchestrator.py         # Per-request EvictionOrchestrator
│   │   ├── segmenter.py            # ThoughtSegmenter, ThoughtSegment
│   │   ├── strategies.py           # GlobalStrategy, ThoughtMin/Avg, RandomStrategy
│   │   └── block_utils.py          # merge_overlapping_ranges, align_ranges_to_blocks, apply_retention_window
│   ├── v1/                         # vLLM v1 engine (modified in this fork)
│   │   ├── attention/
│   │   │   ├── l2_norm_cache.py    # L2NormCache singleton, RequestL2NormData (new)
│   │   │   ├── backend.py          # Attention backend base
│   │   │   └── backends/
│   │   │       └── flash_attn.py   # Flash attention backend (modified: compute_l2_norms flag)
│   │   ├── core/
│   │   │   └── sched/
│   │   │       ├── scheduler.py    # Modified: update_request_mask, _process_evictions, norm retrieval
│   │   │       └── output.py       # Modified: SchedulerOutput.evictable_token_ranges_map
│   │   ├── engine/
│   │   │   ├── async_llm.py        # Modified: update_request_mask, get_request_l2_norms methods
│   │   │   ├── core.py             # Modified: update_request_mask, configure_l2_norms
│   │   │   ├── core_client.py      # Modified: update_request_mask_async, get_request_l2_norms_async
│   │   │   ├── output_processor.py # Modified: propagates new_l2_norms into RequestOutput
│   │   │   └── __init__.py         # Modified: EngineCoreOutput.new_l2_norms field
│   │   └── worker/
│   │       ├── gpu_model_runner.py # Modified: _compute_l2_norms(), block invalidation, evicted_ranges
│   │       ├── gpu_worker.py       # Modified: get_request_l2_norms, configure_l2_norms
│   │       └── worker_base.py      # Modified: eviction-related worker base methods
│   ├── entrypoints/
│   │   └── openai/
│   │       └── chat_completion/
│   │           ├── serving.py      # Modified: EvictionOrchestrator instantiation, stream wrapping
│   │           └── protocol.py     # Modified: EvictionParams, ChatCompletionRequest.eviction_params
│   ├── outputs.py                  # Modified: RequestOutput.new_l2_norms field
│   ├── sampling_params.py          # Modified: enable_l2_norms, l2_norm_layers fields
│   ├── sequence.py                 # Core sequence data structures
│   ├── config/                     # Model, cache, server configuration
│   ├── engine/                     # Legacy engine (v0)
│   ├── model_executor/             # Model implementations
│   ├── attention/                  # Attention abstractions
│   ├── distributed/                # Distributed inference
│   ├── lora/                       # LoRA adapters
│   ├── reasoning/                  # Reasoning model utilities
│   └── logits_process.py           # Logits processors
├── tests/
│   ├── eviction/                   # Eviction-specific tests (new in this fork)
│   │   ├── __init__.py
│   │   ├── test_block_utils.py
│   │   ├── test_strategies.py
│   │   ├── test_segmenter.py
│   │   ├── test_orchestrator.py
│   │   ├── test_protocol_extension.py
│   │   ├── test_serving_integration.py
│   │   ├── test_l2_norm_delivery.py
│   │   ├── test_no_eviction_guard.py
│   │   └── test_scheduler_eviction_fix.py
│   └── [upstream vLLM test dirs]   # v1/, models/, entrypoints/, etc.
├── csrc/                           # C++/CUDA kernel sources (upstream, not modified)
├── benchmarks/                     # Benchmark scripts
├── docs/                           # Documentation
├── examples/                       # Usage examples
├── requirements/                   # Pinned dependencies
├── eviction_changes.patch          # Git patch capturing all fork changes
├── pyproject.toml                  # Build system configuration
└── setup.py                        # Package build
```

## Directory Purposes

**`vllm/thought_eviction/`:**
- Purpose: All custom eviction logic introduced by this fork; self-contained module with no upstream dependencies except `vllm.logger` and `vllm.outputs`
- Contains: Orchestrator, segmenter, four strategy classes, block utilities
- Key files: `orchestrator.py`, `segmenter.py`, `strategies.py`, `block_utils.py`

**`vllm/v1/attention/`:**
- Purpose: Attention layer implementations for the v1 engine; extended with `l2_norm_cache.py` to store per-request L2 norm buffers computed during attention forward passes
- Key files: `l2_norm_cache.py` (new), `backends/flash_attn.py` (modified)

**`vllm/v1/core/sched/`:**
- Purpose: v1 scheduler and its output data structures; modified to accept `update_request_mask` commands and carry `evictable_token_ranges_map` to the worker
- Key files: `scheduler.py`, `output.py`

**`vllm/v1/engine/`:**
- Purpose: Async engine coordination layer (API process side); exposes `update_request_mask` and `get_request_l2_norms` as async utility RPCs
- Key files: `async_llm.py`, `core.py`, `core_client.py`, `output_processor.py`, `__init__.py`

**`vllm/v1/worker/`:**
- Purpose: GPU worker process; runs the forward pass, computes L2 norms post-forward, performs physical KV block invalidation
- Key files: `gpu_model_runner.py`, `gpu_worker.py`

**`vllm/entrypoints/openai/chat_completion/`:**
- Purpose: OpenAI-compatible chat completion HTTP handler; modified to instantiate `EvictionOrchestrator`, wrap the streaming generator, and emit final eviction stats
- Key files: `serving.py`, `protocol.py`

**`tests/eviction/`:**
- Purpose: Unit and integration tests for all thought eviction components; isolated from upstream vLLM tests
- Contains: Per-module unit tests, integration tests for the serving layer, scheduler eviction fix tests

## Key File Locations

**Entry Points:**
- `vllm/entrypoints/openai/chat_completion/serving.py`: Chat completion handler; eviction wiring starts at `create_chat_completion()`

**Configuration / Parameters:**
- `vllm/entrypoints/openai/chat_completion/protocol.py`: `EvictionParams` Pydantic model (lines ~160-177), `ChatCompletionRequest.eviction_params` field (line ~370)
- `vllm/sampling_params.py`: `enable_l2_norms: bool = False`, `l2_norm_layers: list[int] | None = None` (lines ~226-233)

**Core Eviction Logic:**
- `vllm/thought_eviction/orchestrator.py`: `EvictionOrchestrator` — lifecycle owner for a single eviction-enabled request
- `vllm/thought_eviction/segmenter.py`: `ThoughtSegmenter`, `ThoughtSegment` dataclass
- `vllm/thought_eviction/strategies.py`: All four strategy classes
- `vllm/thought_eviction/block_utils.py`: Three pure utility functions

**L2 Norm Infrastructure:**
- `vllm/v1/attention/l2_norm_cache.py`: `L2NormCache` singleton, `get_l2_norm_cache()`
- `vllm/v1/worker/gpu_model_runner.py`: `_compute_l2_norms()` method (~line 1225), block invalidation (~line 1085)

**IPC Data Structures:**
- `vllm/v1/engine/__init__.py`: `EngineCoreOutput.new_l2_norms` field (~line 150)
- `vllm/v1/core/sched/output.py`: `SchedulerOutput.evictable_token_ranges_map` field (~line 242)
- `vllm/outputs.py`: `RequestOutput.new_l2_norms` field (~line 125)

**Testing:**
- `tests/eviction/`: All eviction-specific tests

## Naming Conventions

**Files:**
- Snake_case module names: `thought_eviction/`, `l2_norm_cache.py`, `block_utils.py`
- Test files prefixed with `test_`: `test_strategies.py`, `test_segmenter.py`

**Classes:**
- PascalCase: `EvictionOrchestrator`, `ThoughtSegmenter`, `ThoughtSegment`, `GlobalStrategy`, `L2NormCache`, `EvictionParams`

**Functions/Methods:**
- Snake_case: `compute_evictable_ranges`, `update_norms_batch`, `wrap_stream`, `build_eviction_payload`
- Private methods prefixed with `_`: `_run_eviction_cycle`, `_accumulate`, `_maybe_schedule_cycle`, `_build_strategy`, `_indices_to_ranges`

**Module-level constants:**
- UPPER_SNAKE_CASE: `MAX_SEQ_LEN`, `TARGET_PHRASES`, `_SEPARATOR_OVERLAP`, `_OVERLAP_TOKENS`

**IPC / data structure fields:**
- Snake_case: `new_l2_norms`, `evictable_token_ranges_map`, `enable_l2_norms`, `l2_norm_layers`

## Where to Add New Code

**New Eviction Strategy:**
- Implementation: `vllm/thought_eviction/strategies.py` — add a new class with `compute_evictable_ranges()` method matching existing signatures
- Register: Add a branch in `_build_strategy()` in `vllm/thought_eviction/orchestrator.py`
- Type guard: Add the new name to the `Literal[...]` type in `EvictionParams.strategy` in `vllm/entrypoints/openai/chat_completion/protocol.py`
- Tests: `tests/eviction/test_strategies.py`

**New Eviction Parameter:**
- Add field to `EvictionParams` in `vllm/entrypoints/openai/chat_completion/protocol.py`
- Consume it in `EvictionOrchestrator.__init__()` via `self.params` in `vllm/thought_eviction/orchestrator.py`

**New L2 Norm Layer Filter / Aggregation Logic:**
- Modify `L2NormCache.update_norms_batch()` in `vllm/v1/attention/l2_norm_cache.py`
- Update `GPUModelRunner._compute_l2_norms()` in `vllm/v1/worker/gpu_model_runner.py` if the computation shape changes

**New Block Utility Function:**
- Add to `vllm/thought_eviction/block_utils.py`
- Import and call from `EvictionOrchestrator._run_eviction_cycle()`
- Tests: `tests/eviction/test_block_utils.py`

**New Test for Eviction:**
- Location: `tests/eviction/test_<module>.py`
- Follow existing pattern: unit tests mock the tokenizer and engine client; no live GPU required

## Special Directories

**`vllm/thought_eviction/`:**
- Purpose: The entire custom eviction feature; all new Python added by this fork lives here (except modifications to existing v1 files)
- Generated: No
- Committed: Yes

**`vllm/__pycache__/`, `tests/eviction/__pycache__/`:**
- Purpose: Python bytecode cache
- Generated: Yes
- Committed: No (but present in working tree)

**`vllm.egg-info/`:**
- Purpose: Installed package metadata from `pip install -e .`
- Generated: Yes
- Committed: No

**`eviction_changes.patch`:**
- Purpose: Git patch capturing all fork-specific changes relative to upstream vLLM; useful for rebasing or reviewing the full diff
- Generated: No (manually maintained)
- Committed: Yes

---

*Structure analysis: 2026-04-07*
