# Stack Research: vLLM v0.14 → v0.19 Upgrade

**Domain:** vLLM fork upgrade — dependency and build system changes
**Researched:** 2026-04-07
**Confidence:** HIGH — all findings from verified local git blobs, tag objects, and commit history

---

## Summary

The vLLM v0.14.0rc2 → v0.19.0 upgrade is **not a dependency migration problem**. Python, PyTorch, FlashInfer, CUDA, and build system versions are all identical between the two versions in this repository. The entire upgrade challenge is an **internal API change problem** — vLLM's own module structure, IPC protocols, and component interfaces changed significantly while external dependencies remained pinned.

---

## Dependency Version Table (Before → After)

| Dependency | v0.14.0rc2 | v0.19.0 target | Status |
|------------|-----------|----------------|--------|
| Python | `>=3.10,<3.14` | Same | No change |
| `torch` | `==2.9.1` | `==2.9.1` | No change — pinned in pyproject.toml at both tags |
| `flashinfer-python` | `==0.5.3` | `==0.5.3` | No change |
| `ray[cgraph]` | `>=2.48.0` | Same or higher floor | Verify on fetch |
| `grpcio` | `>=1.76.0` | Same or higher floor | Verify on fetch |
| `transformers` | `>=4.56.0,<5` | Same or higher floor | Verify on fetch |
| `cmake` | `>=3.26.1` | Same | No change |
| `anthropic` | `==0.71.0` (pinned) | `>=0.71.0` (floor) | Relaxed, non-breaking |
| CUDA | 12.x / 13.x | Same | No change |
| `pydantic` | `>=2.12.0` | Same | No change |
| `pyzmq` | `>=25.0.0` | Same | No change |
| `msgspec` | present | present | No change |

---

## Build System Changes

**None expected.** `pyproject.toml`, `setup.py`, `CMakeLists.txt`, and `requirements/build.txt` structure are identical at v0.14.0rc2. The gRPC proto compilation path in `setup.py` is unchanged. The eviction changes do not add any new proto files — only a new Python `EngineCoreRequestType` enum value.

---

## Internal API Changes That Will Break During Upgrade

These are the actual upgrade hazards — not dependency issues:

| Component | v0.14.0rc2 state | Expected v0.19.0 state | Eviction Impact |
|-----------|-----------------|----------------------|-----------------|
| `vllm/v1/worker/gpu_model_runner.py` | Present; eviction hooks here | Substantially refactored, subcomponents extracted | CRITICAL — every hunk needs manual placement |
| `vllm/v1/worker/gpu/model_runner.py` | Present (V2, coexisting) | May be primary model runner | May need eviction hooks migrated here |
| `vllm.v1.engine.processor.Processor` | Shim module | Deleted | LOW — eviction uses `InputProcessor` directly |
| `AsyncLLM.processor` property | Deprecated but present | Removed | None — eviction doesn't use it |
| `EngineCoreRequestType` enum | Ends at `b'\x04'` | May have new values | Eviction adds `UPDATE_MASK = b'\x05'` — check for collision |
| `EngineCoreOutput` struct | Existing fields | May have new fields added | Eviction appends `new_l2_norms` — field ordering matters for msgpack |
| `vllm/v1/core/sched/scheduler.py` | Eviction adds two dicts + two methods | Ubatch scheduling, new return type | High conflict probability |
| `vllm/v1/engine/core_client.py` | Eviction adds `update_request_mask*` methods | DP-aware routing added | Medium conflict probability |
| `vllm/v1/engine/async_llm.py` | Eviction modifies to wire l2_norms | DP support added | Medium conflict probability |

---

## Files the Eviction Patch Touches (from `eviction_changes.patch`)

All of these require manual conflict resolution against v0.19.0:

- `vllm/v1/worker/gpu_model_runner.py` — L2 norm cache init, KV replacement strategy dispatch
- `vllm/v1/worker/gpu_worker.py` — collective RPC for `get_request_l2_norms`
- `vllm/v1/worker/worker_base.py` — RPC method registration
- `vllm/v1/engine/__init__.py` — `EngineCoreOutput.new_l2_norms`, `EngineCoreRequestType.UPDATE_MASK`
- `vllm/v1/engine/core.py` — `update_request_mask()`, `get_request_l2_norms()` methods
- `vllm/v1/engine/core_client.py` — async/sync mask update delegation
- `vllm/v1/engine/async_llm.py` — l2_norms propagation to output stream
- `vllm/v1/engine/input_processor.py` — enable_l2_norms flag wiring
- `vllm/v1/engine/output_processor.py` — new_l2_norms field handling
- `vllm/v1/core/sched/scheduler.py` — eviction data storage, `_process_evictions()`
- `vllm/v1/core/sched/output.py` — SchedulerOutput additions for eviction
- `vllm/v1/core/kv_cache_manager.py` — block freeing for eviction
- `vllm/v1/core/kv_cache_coordinator.py` — eviction coordinator hooks
- `vllm/v1/core/single_type_kv_cache_manager.py` — single-type eviction support
- `vllm/v1/attention/backends/flash_attn.py` — L2 norm hook in attention kernel
- `vllm/sampling_params.py` — `enable_l2_norms: bool`, `l2_norm_layers: list[int] | None`
- `vllm/entrypoints/openai/chat_completion/protocol.py` — `EvictionParams` Pydantic model
- `vllm/entrypoints/openai/chat_completion/serving.py` — eviction params wiring
- `vllm/entrypoints/openai/api_server.py` — `/v1/attention/l2_norms` endpoint registration
- `vllm/entrypoints/api_server.py` — same, plus `UpdateMaskRequest` handling
- `vllm/outputs.py` — `RequestOutput` l2_norms field

---

## Open Questions (Require `git fetch upstream` to Resolve)

1. **Is `vllm/v1/worker/gpu_model_runner.py` the primary runner at v0.19.0?** If the V2 runner at `gpu/model_runner.py` replaced it, eviction hooks must be migrated there.
2. **Did upstream add `EngineCoreRequestType` values at `b'\x05'` or above** between v0.15 and v0.19?
3. **What new fields were added to `EngineCoreOutput`** between v0.14 and v0.19? The eviction `new_l2_norms` field must be added after all upstream new fields.
4. **What changed in `scheduler.py` between v0.14 and v0.19?** Specifically: the return type of `update_from_output` and the structure of `schedule()`.

---

*Research date: 2026-04-07*
