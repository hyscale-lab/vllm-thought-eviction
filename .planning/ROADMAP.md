# Roadmap: vLLM v0.14 → v0.19 Upgrade (Thought Eviction)

## Overview

Upgrade the hyscale-lab/vllm-thought-eviction fork from v0.14.0rc2 to v0.19.0 by manually re-applying all custom thought eviction changes onto the v0.19 base. Work proceeds on the `upgrade_vllm` branch in four phases: pre-upgrade groundwork and audit, safe file/field additions, high-risk structural adaptation (scheduler + GPU runner), and final serving wiring with full validation.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Groundwork & Audit** - Fix pre-existing bugs, write functional smoke test, resolve 5 open research questions
- [ ] **Phase 2: Safe Additions** - Copy eviction modules to v0.19 base, append fields to structs, extend interface/IPC layer
- [ ] **Phase 3: Core Adaptation** - Re-apply scheduler logic with return-type rewrite, adapt GPU model runner to v0.19 block table API
- [ ] **Phase 4: Serving & Validation** - Wire serving layer, run full eviction test suite, verify live server handles eviction requests

## Phase Details

### Phase 1: Groundwork & Audit
**Goal**: The codebase is free of pre-existing crashes and all structural unknowns about v0.19 are resolved before any merge work begins
**Depends on**: Nothing (first phase)
**Requirements**: CLEAN-01, CLEAN-02, CLEAN-03, AUDIT-01, AUDIT-02, AUDIT-03, AUDIT-04, AUDIT-05
**Success Criteria** (what must be TRUE):
  1. `api_server.py` no longer crashes on startup due to Pydantic v1 `parse_obj` call
  2. `InprocClient.update_request_mask_async` exists and is callable
  3. A functional smoke test drives the full eviction pipeline end-to-end and produces an observable pass/fail result
  4. `EngineCoreRequestType` byte values in v0.19 are documented and `b'\x05'` collision status is confirmed
  5. SamplingParams field additions, block table API, schedule() structure, and all EngineCoreClient subclasses in v0.19 are enumerated and recorded
**Plans**: 3 plans
Plans:
- [x] 01-01-PLAN.md -- Fix crash bugs (CLEAN-01, CLEAN-02) and cleanup dead code (D-11 through D-15)
- [x] 01-02-PLAN.md -- Audit v0.19.0 internals (AUDIT-01 through AUDIT-05)
- [x] 01-03-PLAN.md -- Write functional smoke test (CLEAN-03)

### Phase 2: Safe Additions
**Goal**: All eviction modules, custom fields, and IPC interface extensions exist in the v0.19 codebase with zero conflicts — the eviction code is present but not yet wired into runtime logic
**Depends on**: Phase 1
**Requirements**: MERGE-01, MERGE-02, MERGE-03, MERGE-04, MERGE-05, MERGE-06, MERGE-07, MERGE-08, IPC-01, IPC-02, IPC-03, IPC-04, IPC-05
**Success Criteria** (what must be TRUE):
  1. `vllm/thought_eviction/` module, `l2_norm_cache.py`, and `tests/eviction/` are present in the v0.19 tree
  2. `SamplingParams` carries `enable_l2_norms` and `l2_norm_layers` as the last fields; `EngineCoreOutput` carries `new_l2_norms` as the last field; `SchedulerOutput` carries `evictable_token_ranges_map`; `RequestOutput` carries `new_l2_norms`
  3. `UPDATE_MASK` exists in `EngineCoreRequestType` at a non-colliding byte value
  4. `WorkerBase` ABC declares eviction abstract methods; `GPUWorker` implements them; all v0.19 `EngineCoreClient` subclasses (including `DPAsyncMPClient`) expose the eviction IPC methods
  5. `AsyncLLM` exposes `update_request_mask` and `get_request_l2_norms`; `EngineCore`/`EngineCoreProc` dispatch `UPDATE_MASK` correctly
**Plans**: 2 plans
Plans:
- [x] 02-01-PLAN.md -- Reconcile eviction modules and struct fields (MERGE-01..08)
- [x] 02-02-PLAN.md -- Reconcile IPC layer: worker ABC, clients, engine dispatch (IPC-01..05)

### Phase 3: Core Adaptation
**Goal**: The scheduler and GPU model runner correctly execute thought eviction logic on the v0.19 runtime — eviction requests are processed, L2 norms are computed, and token ranges are tracked
**Depends on**: Phase 2
**Requirements**: SCHED-01, SCHED-02, SCHED-03, SCHED-04, SCHED-05, SCHED-06, GPU-01, GPU-02, GPU-03, GPU-04, GPU-05
**Success Criteria** (what must be TRUE):
  1. Scheduler carries `request_eviction_data` and `_l2_norm_last_index` state and `update_request_mask` / `_process_evictions` methods execute without error on a live request
  2. L2 norm retrieval in `update_from_output` correctly handles the v0.19 multi-client dict return type (no AttributeError or silent empty result)
  3. KV cache block freeing works through the v0.19 three-class coordinator hierarchy (no crash on eviction)
  4. All `SchedulerOutput` construction sites pass `evictable_token_ranges_map` (no silent empty-dict regression)
  5. GPU model runner computes L2 norms using the v0.19 `InputBatch.block_table` API without crash; dead `FlashAttentionMetadata` fields are removed
**Plans**: TBD

### Phase 4: Serving & Validation
**Goal**: The upgraded server starts, accepts chat completion requests with `eviction_params`, returns L2 norms, and all existing eviction tests pass
**Depends on**: Phase 3
**Requirements**: SERVE-01, SERVE-02, SERVE-03, SERVE-04, SERVE-05, SERVE-06
**Success Criteria** (what must be TRUE):
  1. `ChatCompletionRequest` accepts `eviction_params` field and routes it through `create_chat_completion` without error
  2. `/v1/attention/l2_norms` endpoint is registered and returns data for an active request
  3. All tests in `tests/eviction/` pass
  4. Server starts on v0.19.0 base without error
  5. A live chat completion request with `eviction_params` completes successfully and the functional smoke test passes end-to-end
**Plans**: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Groundwork & Audit | 3/3 | Complete | - |
| 2. Safe Additions | 0/2 | Planning complete | - |
| 3. Core Adaptation | 0/TBD | Not started | - |
| 4. Serving & Validation | 0/TBD | Not started | - |
