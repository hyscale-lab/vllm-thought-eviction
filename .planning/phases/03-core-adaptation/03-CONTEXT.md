# Phase 3: Core Adaptation - Context

**Gathered:** 2026-04-08
**Status:** Ready for planning

<domain>
## Phase Boundary

Re-apply eviction logic to the scheduler and GPU model runner — port the pre-merge eviction code (commit `2ec9a65e84b`) and adapt it to v0.19's APIs. Both scheduler and GPU runner are pure upstream after Phase 2.1 (Pattern B wholesale accept). Phase 3 adds back all eviction dicts, methods, integration points, and updates existing tests to match the new code.

**Correction from original context:** Phase 3 was originally described as "pure verification." This was wrong — the Phase 2.1 merge stripped all eviction code from Pattern B files. Phase 3 is active implementation: adding eviction code back, adapted to v0.19 APIs.

</domain>

<decisions>
## Implementation Decisions

### Adaptation Approach
- **D-01:** Port from pre-merge reference (commit `2ec9a65e84b`), adapting each piece to v0.19 APIs. Preserves proven logic, just updates API calls.
- **D-02:** Both scheduler AND GPU model runner adaptation in Phase 3 — they're tightly coupled (scheduler sends eviction ranges, GPU runner uses them). Natural unit.
- **D-03:** Verification level: import + unit tests. All eviction modules import cleanly, existing unit tests pass with updated expectations, key classes instantiate without error.

### API Adaptation — Native v0.19 Patterns
- **D-04:** Native adaptation — rewrite eviction code to use v0.19 APIs directly. No shims or wrappers.
- **D-05:** L2 norm retrieval inside the existing `update_from_output` output loop. Retrieve from `L2NormCache` while building each `EngineCoreOutput`, adapted to the `dict[int, EngineCoreOutputs]` return structure (keyed by `engine_index`).
- **D-06:** Block table invalidation targets KV cache group 0 only via `MultiGroupBlockTable[0]`. Pre-merge eviction was designed for single-group; no multi-group expansion.
- **D-07:** KV cache block freeing calls `self.kv_cache_coordinator.free_blocks(request_id, block_indices)` — method already exists in upstream.

### Scheduler — What to Add
- **D-08:** Add `request_eviction_data: dict[str, list[tuple[int, int]]]` and `_l2_norm_last_index: dict[str, int]` to `Scheduler.__init__`
- **D-09:** Add `update_request_mask()` method — receives eviction ranges from IPC, stores in `request_eviction_data`
- **D-10:** Add `_process_evictions()` method — reads `request_eviction_data`, calls `kv_cache_coordinator.free_blocks()`, populates `evictable_token_ranges_map` on `SchedulerOutput`
- **D-11:** Add `_process_evictions()` call in `schedule()` method
- **D-12:** Add L2 norm retrieval in `update_from_output` — differential retrieval from `L2NormCache`, attach `new_l2_norms` to `EngineCoreOutput`
- **D-13:** Add eviction cleanup in `_free_request()` — remove request from `request_eviction_data` and `_l2_norm_last_index`

### GPU Model Runner — What to Add
- **D-14:** Add `_compute_l2_norms()` method — compute mean L2 norms from key cache across selected layers, write to `L2NormCache`
- **D-15:** Add L2 norm cache initialization in `__init__`: `self.l2_norm_cache = get_l2_norm_cache()`
- **D-16:** Add block table invalidation for eviction ranges from `scheduler_output.evictable_token_ranges_map`, using `self.input_batch.block_table[0]`
- **D-17:** Add `_compute_l2_norms()` call in `execute_model` for requests with `enable_l2_norms=True`
- **D-18:** Add request cleanup: `self.l2_norm_cache.remove_request(req_id)` on request finish

### Test Strategy
- **D-19:** Update source-inspect tests to match new v0.19-adapted function signatures and bodies. They verify eviction additions are present in the adapted code.
- **D-20:** Run existing `tests/eviction/` suite only — no new mock integration tests. If existing tests pass with updated expectations, eviction code is verified.
- **D-21:** Functional smoke test (full server) stays in Phase 4 scope.

### Claude's Discretion
- Exact insertion points for eviction code in scheduler.py and gpu_model_runner.py
- Commit granularity within Phase 3 plans
- Plan count and ordering (scheduler-first vs GPU-first vs parallel)
- Whether to add lightweight integration checks beyond import + unit tests
- Specific adaptations needed for each pre-merge function when porting to v0.19

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Pre-Merge Reference (source for porting)
- Pre-merge commit `2ec9a65e84b` — Working eviction code in scheduler and GPU runner before upstream merge stripped it. Use `git show 2ec9a65e84b:path/to/file` to retrieve.

### Phase 2.1 Merge Records
- `.planning/phases/02.1-upstream-merge/02.1-CONTEXT.md` — Merge decisions, Pattern A/B classification
- `.planning/phases/02-safe-additions/02-CONTEXT.md` — Phase 2 reconciliation decisions, field positioning

### Phase 1 Audit Findings
- `.planning/phases/01-groundwork-audit/01-AUDIT-FINDINGS.md` — v0.19 structural analysis: byte values, field layout, block table API, schedule() structure, client hierarchy

### Scheduler (adaptation targets)
- `vllm/v1/core/sched/scheduler.py` — Pure upstream, 2317 lines. All eviction code to be added here.
- `vllm/v1/core/sched/output.py` — `SchedulerOutput.evictable_token_ranges_map` already exists (line 243)

### GPU Model Runner (adaptation targets)
- `vllm/v1/worker/gpu_model_runner.py` — Pure upstream, 7059 lines. All eviction code to be added here.
- `vllm/v1/worker/block_table.py` — `MultiGroupBlockTable` class, `BlockTable` class

### KV Cache Coordinator
- `vllm/v1/core/kv_cache_coordinator.py` — `free_blocks()` at line 191, delegates to `single_type_managers`

### Engine Core Output (L2 norm target)
- `vllm/v1/engine/__init__.py` — `EngineCoreOutput` struct with `new_l2_norms` field, `EngineCoreOutputs` wrapper

### L2 Norm Cache
- `vllm/v1/attention/l2_norm_cache.py` — Singleton cache, `get_l2_norm_cache()`, differential retrieval

### Eviction Module (already present)
- `vllm/thought_eviction/` — All eviction module files (orchestrator, segmenter, strategies, block_utils)

### Tests
- `tests/eviction/` — All existing eviction test files (source-inspect and functional)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- Pre-merge commit `2ec9a65e84b` — Complete working eviction code for scheduler and GPU runner
- Phase 2 reconciled IPC layer — all client methods already wired and callable
- `SchedulerOutput.evictable_token_ranges_map` field already present (auto-merged)

### Established Patterns
- `import regex as re` required (not `import re`) — enforced by pre-commit
- `update_from_output` returns `dict[int, EngineCoreOutputs]` keyed by `engine_index`
- Block table accessed via `self.input_batch.block_table[kv_cache_gid]` (MultiGroupBlockTable)
- KV cache coordinator: `self.kv_cache_coordinator.free_blocks(request_id, block_indices)`
- EngineCoreOutput constructed per-request inside the output loop (line 1464)

### Integration Points
- Scheduler `__init__` — add eviction dicts after existing state initialization
- Scheduler `schedule()` — add `_process_evictions()` call at appropriate point
- Scheduler `update_from_output` output loop (line 1464) — add L2 norm retrieval
- Scheduler `_free_request()` — add eviction state cleanup
- GPU runner `__init__` — add L2 norm cache initialization
- GPU runner `execute_model` — add L2 norm computation call and block invalidation

</code_context>

<specifics>
## Specific Ideas

- The pre-merge code used a flat block table; v0.19 uses `MultiGroupBlockTable[0]` for group 0 — direct index access
- `update_from_output` return type changed from flat to `dict[int, EngineCoreOutputs]` — L2 norms must be attached inside the per-request loop, keyed correctly
- `free_blocks` signature matches: `free_blocks(request_id, block_indices)` — same pattern, just accessed via coordinator now

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 03-core-adaptation*
*Context gathered: 2026-04-08*
