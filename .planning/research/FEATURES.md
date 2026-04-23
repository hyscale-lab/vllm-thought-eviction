# Features Research: vLLM v0.14 → v0.19 Custom Integration Points

**Domain:** vLLM fork upgrade — thought eviction feature compatibility
**Researched:** 2026-04-07
**Confidence:** MEDIUM — based on direct diff analysis of the 45-file eviction patch and current fork source. Upstream v0.15–v0.19 changelogs were not fetchable at research time; v0.15–v0.19 tags are not present locally. Risk assessments are inferred from what the fork modifies and which vLLM internals have high churn rates.

---

## Summary

The thought eviction feature set is a 45-file, ~5,600-line patch that hooks into vLLM's most volatile internals: the scheduler loop, the GPU model runner, the IPC envelope structs, and the KV cache manager. Two structural hazards dominate: (1) msgspec positional encoding on `array_like=True` structs that silently corrupts IPC if field ordering is wrong, and (2) the scheduler's `update_from_output` return type changing in v0.19, requiring logic adaptation — not just re-patching.

---

## Custom Integration Points — Risk Assessment

| Integration Point | File | Risk | Why |
|------------------|------|------|-----|
| `enable_l2_norms`, `l2_norm_layers` fields | `sampling_params.py` | HIGH | Positional msgspec encoding; upstream fields added between v0.14–v0.19 shift positions |
| `new_l2_norms` field | `vllm/v1/engine/__init__.py` | HIGH | Same positional hazard on `EngineCoreOutput` |
| `UPDATE_MASK = b'\x05'` | `vllm/v1/engine/__init__.py` | MEDIUM | Byte collision if upstream added new request types in this range |
| `_handle_client_request` UPDATE_MASK dispatch | `vllm/v1/engine/core.py` | HIGH | Method heavily modified by DP/ubatch additions in v0.19 |
| New utility methods on `EngineCore` | `vllm/v1/engine/core.py` | LOW | Pure additions; unlikely to conflict |
| `update_request_mask_async` on `AsyncMPClient` | `vllm/v1/engine/core_client.py` | HIGH | `core_client.py` restructured for DP routing; new subclasses not covered |
| UUID suffix removal in `assign_request_id` | `vllm/v1/engine/input_processor.py` | HIGH | Line will have moved; pre-existing correctness bug to be careful with |
| `new_l2_norms` threading in `output_processor` | `vllm/v1/engine/output_processor.py` | MEDIUM | Additive parameter; signature may have grown |
| `evictable_token_ranges_map` on `SchedulerOutput` | `vllm/v1/core/sched/output.py` | HIGH | `SchedulerOutput` is high-churn; must re-append after all v0.19 fields |
| `_process_evictions()` prologue in `schedule()` | `vllm/v1/core/sched/scheduler.py` | HIGH | Ubatch restructuring moved the injection point |
| L2 norm retrieval in `update_from_output` | `vllm/v1/core/sched/scheduler.py` | HIGH | Return type changed to `dict[int, EngineCoreOutputs]`; logic must be adapted |
| `free_blocks` on KV cache classes | `kv_cache_manager.py`, `coordinator.py`, `single_type_kv_cache_manager.py` | MEDIUM | Class hierarchy refactored; call sites must be rerouted |
| Block-table invalidation in `_update_states` | `vllm/v1/worker/gpu_model_runner.py` | HIGH | Internal `block_tables`/`.np` API access; likely gone in v0.19 |
| `_compute_l2_norms` method | `vllm/v1/worker/gpu_model_runner.py` | MEDIUM | Pure addition; reference to `attn_metadata_dict` variable name must be verified |
| `execute_model` L2 norm call | `vllm/v1/worker/gpu_model_runner.py` | MEDIUM | Context lines around `_model_forward` call will have shifted |
| `compute_l2_norms`, `request_ids` fields | `vllm/v1/attention/backends/flash_attn.py` | MEDIUM | Recommend removal — never read by kernel (confirmed in CONCERNS.md) |
| `eviction_params` on `ChatCompletionRequest` | `vllm/entrypoints/openai/chat_completion/protocol.py` | LOW | Additive field; protocol files are stable |
| Eviction wiring in `create_chat_completion` | `vllm/entrypoints/openai/chat_completion/serving.py` | MEDIUM | serving.py evolves but injection is clean |
| Attention router registration | `vllm/entrypoints/openai/api_server.py` | MEDIUM | Router registration section may be restructured |
| `parse_obj` → Pydantic v2 bug | `vllm/entrypoints/api_server.py` | HIGH | Pre-existing crash (Pydantic v1 API); will persist through upgrade and appear as regression |

---

## Breaking API Changes with Migration Paths

### SamplingParams (msgspec positional encoding)
- Before: Fork appends `enable_l2_norms: bool = False` and `l2_norm_layers: list[int] | None = None`
- Risk: Upstream fields inserted before these in v0.19 shift positional encoding
- Migration: Obtain v0.19 `sampling_params.py`; append both eviction fields after all upstream fields; write a msgspec round-trip test

### EngineCoreOutput (msgspec positional encoding)
- Before: Fork appends `new_l2_norms: list[float] | None = None`
- Migration: Append `new_l2_norms` after all upstream v0.19 fields; same serialization test

### EngineCoreRequestType byte value
- Before: Fork uses `b'\x05'` for `UPDATE_MASK`
- Migration: Inspect v0.19 enum; if `b'\x05'` is taken, use next available byte

### SchedulerOutput dataclass
- Before: Fork appends `evictable_token_ranges_map` field
- Migration: Re-append to end of v0.19 `SchedulerOutput`; verify `make_empty()` does not need explicit init

### schedule() prologue
- Before: `processed_ranges = self._process_evictions()` added at start of `schedule()`
- Migration: Read v0.19 `schedule()` structure; if ubatch loop was added, place `_process_evictions()` at the outermost scheduling scope, not inside ubatch iteration

### update_from_output return type (REQUIRES ADAPTATION, NOT RE-PATCHING)
- Before: Returns flat `EngineCoreOutputs`
- After in v0.19: Returns `dict[int, EngineCoreOutputs]` keyed by client index
- Migration: Rewrite the L2 norm retrieval loop to iterate over the multi-client return dict

### _handle_client_request dispatch
- Before: `elif request_type == EngineCoreRequestType.UPDATE_MASK:` added to dispatch chain
- Migration: Locate the dispatch chain in v0.19; re-insert UPDATE_MASK case; body is unchanged

### FlashAttentionMetadata fields (recommended removal)
- Before: `compute_l2_norms`, `request_ids` on metadata + two build() params added
- Migration: Remove them entirely — zero functional impact since they are never read by the kernel

### Pydantic v1 parse_obj
- Before: `UpdateMaskRequest.parse_obj(json_request)` in `api_server.py`
- Migration: Replace with `UpdateMaskRequest.model_validate(json_request)` — one-line fix, do in Phase 1

### EngineCoreClient subclass coverage
- Before: `update_request_mask*` methods only on `SyncMPClient` and `AsyncMPClient`
- After in v0.19: `DPAsyncMPClient` and actor-based clients added
- Migration: Add the three eviction methods to every new concrete client subclass

---

## Features to Preserve (Must Work After Upgrade)

All of the following are existing, validated requirements:

- Thought eviction with multiple strategies (global, thought_min, thought_avg, random)
- L2 norm computation in GPU worker with IPC propagation
- Per-request `EvictionOrchestrator` middleware
- OpenAI-compatible chat completion API with `eviction_params` extension
- L2 norm polling endpoint (`/v1/attention/l2_norms`)
- Eviction test suite (`tests/eviction/`)

## Features Explicitly Out of Scope

- Refactoring eviction code to use new v0.19.0 APIs beyond what compatibility requires
- Adding new eviction features
- Fixing pre-existing tech debt (dual singleton, UUID suffix issue) beyond Pydantic v1 crash
- Upgrading beyond v0.19.0

---

## New vLLM v0.19 Features That Affect the Upgrade

**Ubatch scheduling:** Restructures `schedule()` body. The eviction prologue must be positioned outside the ubatch iteration to avoid double-processing evictions per step.

**KV Connector framework:** `kv_cache_coordinator.py` and `single_type_kv_cache_manager.py` have a changed class hierarchy. The `free_blocks` addition must be re-located within the v0.19 hierarchy.

**DPAsyncMPClient:** The fork's `update_request_mask_async` only exists on `AsyncMPClient`. In DP mode, eviction masks will not reach all engine shards unless explicitly added to the DP client.

**SchedulerInterface ABC:** The scheduler is now pluggable via `scheduler_config.get_scheduler_cls()`. The fork's eviction methods that modify `Scheduler` directly must be declared on the interface.

---

## Open Questions

1. What exact fields did upstream add to `SamplingParams` between v0.14 and v0.19? (requires `git diff v0.14.0rc2..v0.19.0 -- vllm/sampling_params.py`)
2. Did vLLM add any new `EngineCoreRequestType` byte values in the `b'\x05'` range?
3. What is the exact structure of `schedule()` in v0.19 — is there a ubatch outer loop?
4. What is the v0.19 `InputBatch.block_table` public API — is `.block_tables` still present?

---

*Research date: 2026-04-07*
