# Architecture Research: vLLM v0.14 → v0.19 Structural Changes

**Domain:** vLLM v1 engine internals — structural changes affecting thought eviction
**Researched:** 2026-04-07
**Confidence:** MEDIUM — all findings from live codebase inspection (v0.19 state) and patch diff analysis; upstream changelog not reachable at research time

---

## Summary

Between v0.14 and v0.19, vLLM's v1 engine underwent substantial structural refactoring: the scheduler gained a formal interface and a changed `update_from_output` return type; the GPU model runner had multiple subcomponents extracted to separate files; the KV cache management was split from monolithic into a three-class coordinator hierarchy; and the engine client layer gained new DP-aware concrete subclasses. The eviction patch must be adapted — not just re-applied — to fit the new structure.

---

## v1 Engine: What Was True in v0.14

- `AsyncLLM` as top-level async engine (`vllm/v1/engine/async_llm.py`)
- `EngineCore` / `EngineCoreProc` as inner loop in subprocess (`vllm/v1/engine/core.py`)
- `EngineCoreClient` hierarchy connecting API process to engine subprocess via ZMQ
- `Scheduler` at `vllm/v1/core/sched/scheduler.py` with flat `schedule()` / `update_from_output()` loop
- `GPUModelRunner` at `vllm/v1/worker/gpu_model_runner.py`
- `EngineCoreRequest` / `EngineCoreOutput` / `EngineCoreOutputs` as msgspec/msgpack IPC envelope structs
- `SchedulerOutput` carrying `NewRequestData` and `CachedRequestData` to the worker
- Monolithic KV cache manager at `vllm/v1/core/kv_cache_manager.py`

---

## v1 Engine: What Changed Between v0.14 and v0.19

### Scheduler gained a formal interface

`vllm/v1/core/sched/interface.py` defines `SchedulerInterface` as an ABC. `EngineCore.__init__` now instantiates the scheduler via `vllm_config.scheduler_config.get_scheduler_cls()` — scheduler implementations are pluggable. An `AsyncScheduler` subclass appeared at `vllm/v1/core/sched/async_scheduler.py`. Any code that directly imports `Scheduler` and calls internal methods needs to be reconciled against the interface.

### `update_from_output` return type changed (CRITICAL)

`SchedulerInterface.update_from_output` returns `dict[int, EngineCoreOutputs]` (keyed by client index) instead of a flat `EngineCoreOutputs`. The fork's L2 norm retrieval loop inside `update_from_output` must be adapted to iterate over the new multi-client return structure — not just re-patched.

### KV cache management split into coordinator hierarchy

The monolithic v0.14 KV cache manager is split into three cooperating classes:
- `KVCacheCoordinator` (`vllm/v1/core/kv_cache_coordinator.py`) — abstract base
- `KVCacheManager` (`vllm/v1/core/kv_cache_manager.py`) — block allocation
- `SingleTypeKVCacheManager` (`vllm/v1/core/single_type_kv_cache_manager.py`) — per-spec-type management

The fork calls block-freeing methods in `_process_evictions`. These call sites must be rerouted through the new coordinator API.

### `SchedulerOutput` gained new fields

v0.19 `SchedulerOutput` carries: `preempted_req_ids`, `has_structured_output_requests`, `pending_structured_output_tokens`, `num_invalid_spec_tokens`, `kv_connector_metadata`, `ec_connector_metadata`. The fork's `evictable_token_ranges_map` must be re-appended after all upstream fields.

### `EngineCoreOutputs` gained wave-coordination fields

`wave_complete`, `start_wave`, `finished_requests` added for DP wave coordination. `EngineCoreOutput` gained `routed_experts`, `num_nans_in_logits`, `trace_headers`. The fork's `new_l2_norms: list[float] | None = None` must remain after all upstream fields. The `omit_defaults=True` + `array_like=True` msgspec config makes field ordering critical for serialization.

### Worker reorganized with extracted modules

`gpu_model_runner.py` grew substantially, then had subcomponents extracted:
- `gpu_input_batch.py` — `InputBatch`, `CachedRequestState`
- `gpu_ubatch_wrapper.py` + `ubatching.py` — micro-batching
- `workspace.py` — memory workspace management
- `ec_connector_model_runner_mixin.py`, `kv_connector_model_runner_mixin.py`, `lora_model_runner_mixin.py` — feature mixins

The fork's `_compute_l2_norms`, block invalidation, and `evicted_ranges` logic is in `gpu_model_runner.py`. Every hunk will need manual placement against the refactored file structure.

### Flash attention backend helpers extracted

`flash_attn_varlen_func`, `get_scheduler_metadata`, `reshape_and_cache_flash`, and related helpers were extracted to `backends/fa_utils.py`. The fork's small change to `flash_attn.py` will not apply cleanly because context lines no longer exist at their original positions.

### Engine client layer gained DP-aware subclasses

v0.19 `core_client.py` contains `DPAsyncMPClient` and actor-based clients (`CoreEngineActorManager`, `CoreEngineProcManager`). The fork's eviction methods were added only to `SyncMPClient` and `AsyncMPClient`. Every new concrete subclass needs the three eviction methods.

---

## Module Moves and Renames

| Item | v0.14 location | v0.19 location | Confidence |
|------|---------------|----------------|-----------|
| `OpenAIServing` base | flat `serving_base.py` | `entrypoints/openai/engine/serving.py` | MEDIUM |
| Protocol types (`ErrorResponse` etc.) | flat `protocol.py` | `entrypoints/openai/engine/protocol.py` | HIGH |
| `SchedulerInterface` | did not exist | `v1/core/sched/interface.py` | HIGH |
| `AsyncScheduler` | did not exist | `v1/core/sched/async_scheduler.py` | HIGH |
| KV cache block management | `kv_cache_manager.py` (monolithic) | `kv_cache_coordinator.py` + `kv_cache_manager.py` + `single_type_kv_cache_manager.py` | HIGH |
| `InputBatch` / `CachedRequestState` | inside `gpu_model_runner.py` | `v1/worker/gpu_input_batch.py` | HIGH |
| Flash attention helpers | inside `flash_attn.py` | `v1/attention/backends/fa_utils.py` | HIGH |
| Backend registry | did not exist | `v1/attention/backends/registry.py` | MEDIUM |
| `EngineZmqAddresses` / `EngineHandshakeMetadata` | did not exist | `v1/engine/utils.py` | HIGH |

---

## IPC Mechanism Summary

The ZMQ + msgpack protocol is structurally unchanged. All eviction-specific IPC payloads remain valid in concept; the risk is in the envelope structs' field ordering.

| Direction | Mechanism | Eviction-specific payload |
|-----------|-----------|--------------------------|
| API → Engine (eviction command) | `UPDATE_MASK` message type, ZMQ | `(request_id, evictable_token_ranges)` tuple |
| API → Engine (L2 norm request) | `UTILITY` call via `call_utility_async` | `"get_request_l2_norms", request_id, start_index` |
| Engine → API (L2 norms in output) | `EngineCoreOutput.new_l2_norms`, msgpack | `list[float] | None` (omitted when None) |
| Worker → Engine (L2 norms, in-process) | `L2NormCache` singleton | Thread-safe per-request buffer |

---

## Conflict-Heavy Areas (Ranked)

### 1. `vllm/v1/worker/gpu_model_runner.py` — CRITICAL

Fork adds: `l2_norm_cache` init, `evicted_ranges` dict, `replace_func`/`strategy_map`, block invalidation loop (~60 lines), `_compute_l2_norms` (~40 lines), three KV replacement helpers, finished-request cleanup. Upstream refactored extensively: extracted `InputBatch`, added micro-batching, KV/LoRA/EC mixins, workspace management. Every diff hunk requires manual placement.

### 2. `vllm/v1/engine/core.py` — HIGH

Fork adds: `update_request_mask`, `get_request_l2_norms`, `configure_l2_norms` methods, `UPDATE_MASK` dispatch in `_handle_client_request`. Upstream added: handshake metadata, batch queue for pipeline parallelism, async scheduling flag, GC tuning, elastic EP, KV connector handshake. Both `__init__` and `_handle_client_request` are substantially different.

### 3. `vllm/v1/core/sched/scheduler.py` — HIGH

Fork adds: `request_eviction_data` dict, `_l2_norm_last_index` dict, `update_request_mask` method, `_process_evictions` method, L2 norm retrieval in `update_from_output`. **Critical:** the return type of `update_from_output` changed. The norm retrieval loop needs structural adaptation, not just re-application.

### 4. `vllm/v1/engine/core_client.py` — HIGH

Fork adds: abstract eviction methods on `EngineCoreClient` ABC, concrete implementations on `SyncMPClient` and `AsyncMPClient`. Upstream added DP-aware routing and actor-based clients. Every new concrete subclass needs the eviction methods.

### 5. `vllm/v1/engine/async_llm.py` — MEDIUM

Fork adds `update_request_mask` and `get_request_l2_norms` methods. Upstream added DP support, stat loggers. Low semantic conflict but high context-line conflict.

### 6. `vllm/v1/attention/backends/flash_attn.py` — MEDIUM

Small fork change, but context lines moved to `fa_utils.py`. Patch will not apply cleanly; requires manual re-anchoring. Recommend removing `compute_l2_norms` and `request_ids` fields entirely instead.

### 7. `vllm/v1/core/sched/output.py` — LOW-MEDIUM

Fork appends `evictable_token_ranges_map` field. Upstream added several fields. The field can be re-appended after upstream fields; context-line conflict only.

### 8. `vllm/v1/engine/__init__.py` — LOW

Fork adds `new_l2_norms` to `EngineCoreOutput` and `UPDATE_MASK` to enum. Upstream added fields and enum values. Order must be respected for msgpack serialization.

### 9. `vllm/sampling_params.py` — LOW

Fork appends two fields with defaults. `omit_defaults=True` makes serialization safe if fields are last. Pure context-line conflict.

### 10. `vllm/outputs.py` and serving/protocol files — LOW

Minor kwarg additions and field additions. Very low conflict risk.

---

## Recommended Merge Order

1. **New files first** (zero conflict): `vllm/thought_eviction/`, `vllm/v1/attention/l2_norm_cache.py`, `vllm/entrypoints/openai/extensions/`, `tests/eviction/`
2. **Simple field additions**: `sampling_params.py`, `outputs.py`, `vllm/v1/engine/__init__.py`, `vllm/v1/core/sched/output.py`
3. **Interface extensions**: `worker_base.py`, `gpu_worker.py`, `core_client.py` (add to all subclasses including DP)
4. **Complex engine methods**: `async_llm.py`, `engine/core.py`
5. **Scheduler logic**: `sched/scheduler.py` — requires adaptation for new return type
6. **Model runner**: `gpu_model_runner.py` — requires full manual placement; diff block table API first
7. **Serving integration last**: `api_server.py`, `chat_completion/serving.py`, `protocol.py`, `flash_attn.py`

---

## Critical Adaptation Points (Not Just Re-Patching)

**A. `update_from_output` return type:** Must iterate over `dict[int, EngineCoreOutputs]` (multi-client). The L2 norm population loop inside this method needs structural rewrite.

**B. KV cache block freeing in `_process_evictions`:** Must use `KVCacheCoordinator` API, not direct `kv_cache_manager` attribute (which may not exist in v0.19).

**C. `EngineCoreClient` subclass coverage:** `DPAsyncMPClient` and any actor-based clients in v0.19 must get `update_request_mask`, `update_request_mask_async`, `get_request_l2_norms_async`.

**D. `_handle_client_request` dispatch:** Must locate the dispatch method in v0.19's `EngineCoreProc` (may be renamed/restructured with DP additions) and add the `UPDATE_MASK` branch.

---

*Research date: 2026-04-07*
