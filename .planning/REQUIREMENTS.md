# Requirements: vLLM v0.14 → v0.19 Upgrade

**Defined:** 2026-04-07
**Core Value:** Upgrade to v0.19.0 while preserving working thought eviction

## v1 Requirements

### Pre-Upgrade Cleanup

- [x] **CLEAN-01**: Fix Pydantic v1 `parse_obj` crash in `api_server.py` (replace with `model_validate`)
- [x] **CLEAN-02**: Implement `InprocClient.update_request_mask_async` (missing method)
- [x] **CLEAN-03**: Write functional smoke test that drives full eviction pipeline end-to-end

### Upstream Audit

- [x] **AUDIT-01**: Verify `EngineCoreRequestType` byte values in v0.19 — confirm no collision at `b'\x05'`
- [x] **AUDIT-02**: Identify all new fields added to `SamplingParams` between v0.14 and v0.19
- [x] **AUDIT-03**: Determine v0.19 `InputBatch.block_table` API (is `.block_tables`/`.np` still present?)
- [x] **AUDIT-04**: Map v0.19 `schedule()` structure (ubatch loop layout)
- [x] **AUDIT-05**: Enumerate all v0.19 `EngineCoreClient` subclasses

### New Files and Field Additions

- [x] **MERGE-01**: Copy `vllm/thought_eviction/` module to v0.19 base
- [x] **MERGE-02**: Copy `vllm/v1/attention/l2_norm_cache.py` to v0.19 base
- [x] **MERGE-03**: Copy `tests/eviction/` to v0.19 base
- [x] **MERGE-04**: Append `enable_l2_norms` and `l2_norm_layers` fields to v0.19 `SamplingParams` (after all upstream fields)
- [x] **MERGE-05**: Append `new_l2_norms` field to v0.19 `EngineCoreOutput` (after all upstream fields)
- [x] **MERGE-06**: Add `UPDATE_MASK` to v0.19 `EngineCoreRequestType` (verified non-colliding byte)
- [x] **MERGE-07**: Append `evictable_token_ranges_map` to v0.19 `SchedulerOutput`
- [x] **MERGE-08**: Add `new_l2_norms` field to `RequestOutput` in `outputs.py`

### Interface and IPC Layer

- [x] **IPC-01**: Add eviction abstract methods to v0.19 `WorkerBase` ABC
- [x] **IPC-02**: Implement eviction methods on `GPUWorker`
- [x] **IPC-03**: Add eviction methods to ALL v0.19 `EngineCoreClient` subclasses (including `DPAsyncMPClient` and any actor-based clients)
- [x] **IPC-04**: Add `update_request_mask` and `get_request_l2_norms` to `AsyncLLM`
- [x] **IPC-05**: Add `UPDATE_MASK` dispatch and utility methods to `EngineCore`/`EngineCoreProc`

### Upstream Merge (Phase 02.1 — INSERTED)

- [x] **UPSTREAM-01**: Execute `git merge upstream/main` and resolve all 14 conflicting files
- [x] **UPSTREAM-02**: Resolve Pattern A files (10) by accepting upstream + re-appending eviction additions
- [x] **UPSTREAM-03**: Resolve Pattern B files (4) by accepting upstream wholesale (eviction deferred to Phase 3/4)
- [x] **UPSTREAM-04**: Verify all 7 auto-merged files preserve eviction code correctly
- [x] **UPSTREAM-05**: Validate msgspec field ordering (eviction fields last) in EngineCoreOutput, SchedulerOutput, SamplingParams

### Scheduler Adaptation

- [x] **SCHED-01**: Add `request_eviction_data` and `_l2_norm_last_index` dicts to v0.19 `Scheduler`
- [x] **SCHED-02**: Re-apply `update_request_mask` method on scheduler
- [x] **SCHED-03**: Re-apply `_process_evictions` method with correct placement in v0.19 `schedule()` structure
- [x] **SCHED-04**: Adapt L2 norm retrieval in `update_from_output` for new multi-client dict return type
- [x] **SCHED-05**: Add `free_blocks` support through v0.19 KV cache coordinator hierarchy
- [x] **SCHED-06**: Verify all `SchedulerOutput` construction sites pass `evictable_token_ranges_map`

### GPU Model Runner

- [ ] **GPU-01**: Adapt block table invalidation to v0.19 `InputBatch.block_table` API
- [ ] **GPU-02**: Re-apply `_compute_l2_norms` method in v0.19 model runner
- [ ] **GPU-03**: Re-apply L2 norm cache initialization and KV replacement strategy dispatch
- [ ] **GPU-04**: Re-apply `execute_model` L2 norm computation call
- [ ] **GPU-05**: Remove dead `compute_l2_norms`/`request_ids` fields from `FlashAttentionMetadata` (simplify instead of carry forward)

### Serving and Validation

- [ ] **SERVE-01**: Re-apply `EvictionParams` and `eviction_params` field on `ChatCompletionRequest`
- [ ] **SERVE-02**: Re-apply eviction wiring in `create_chat_completion`
- [ ] **SERVE-03**: Register `/v1/attention/l2_norms` endpoint on v0.19 `api_server.py`
- [ ] **SERVE-04**: All existing eviction tests pass (`tests/eviction/`)
- [ ] **SERVE-05**: Server starts and handles chat completion requests with `eviction_params`
- [ ] **SERVE-06**: Functional smoke test passes end-to-end

## v2 Requirements

### Post-Upgrade Improvements (deferred)

- **DEBT-01**: Fix dual singleton pattern in `L2NormCache`
- **DEBT-02**: Restore UUID suffix in `input_processor.py` for concurrent request safety
- **DEBT-03**: Add functional (not source-inspect) tests for eviction pipeline
- **DEBT-04**: Implement eviction support for DP mode (`DPAsyncMPClient`)

## Out of Scope

| Feature | Reason |
|---------|--------|
| Refactoring eviction code to use new v0.19 patterns | Scope is minimal adaptation only |
| New eviction features | Pure upgrade, no feature work |
| Fixing pre-existing tech debt | Separate effort after upgrade |
| Upgrading beyond v0.19 | One version at a time |
| DP mode eviction support | Complex feature, not needed for upgrade |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| CLEAN-01 | Phase 1 | Complete |
| CLEAN-02 | Phase 1 | Complete |
| CLEAN-03 | Phase 1 | Complete |
| AUDIT-01 | Phase 1 | Complete |
| AUDIT-02 | Phase 1 | Complete |
| AUDIT-03 | Phase 1 | Complete |
| AUDIT-04 | Phase 1 | Complete |
| AUDIT-05 | Phase 1 | Complete |
| MERGE-01 | Phase 2 | Complete |
| MERGE-02 | Phase 2 | Complete |
| MERGE-03 | Phase 2 | Complete |
| MERGE-04 | Phase 2 | Complete |
| MERGE-05 | Phase 2 | Complete |
| MERGE-06 | Phase 2 | Complete |
| MERGE-07 | Phase 2 | Complete |
| MERGE-08 | Phase 2 | Complete |
| IPC-01 | Phase 2 | Complete |
| IPC-02 | Phase 2 | Complete |
| IPC-03 | Phase 2 | Complete |
| IPC-04 | Phase 2 | Complete |
| IPC-05 | Phase 2 | Complete |
| UPSTREAM-01 | Phase 02.1 | Complete |
| UPSTREAM-02 | Phase 02.1 | Complete |
| UPSTREAM-03 | Phase 02.1 | Complete |
| UPSTREAM-04 | Phase 02.1 | Complete |
| UPSTREAM-05 | Phase 02.1 | Complete |
| SCHED-01 | Phase 3 | Complete |
| SCHED-02 | Phase 3 | Complete |
| SCHED-03 | Phase 3 | Complete |
| SCHED-04 | Phase 3 | Complete |
| SCHED-05 | Phase 3 | Complete |
| SCHED-06 | Phase 3 | Complete |
| GPU-01 | Phase 3 | Pending |
| GPU-02 | Phase 3 | Pending |
| GPU-03 | Phase 3 | Pending |
| GPU-04 | Phase 3 | Pending |
| GPU-05 | Phase 3 | Pending |
| SERVE-01 | Phase 4 | Pending |
| SERVE-02 | Phase 4 | Pending |
| SERVE-03 | Phase 4 | Pending |
| SERVE-04 | Phase 4 | Pending |
| SERVE-05 | Phase 4 | Pending |
| SERVE-06 | Phase 4 | Pending |

**Coverage:**
- v1 requirements: 41 total
- Mapped to phases: 41
- Unmapped: 0

---
*Requirements defined: 2026-04-07*
*Last updated: 2026-04-08 — added UPSTREAM-01..05 for Phase 02.1*
