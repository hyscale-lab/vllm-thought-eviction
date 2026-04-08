# Phase 3: Core Adaptation - Context

**Gathered:** 2026-04-08
**Status:** Ready for planning

<domain>
## Phase Boundary

Verify that the scheduler and GPU model runner correctly execute thought eviction logic after the Phase 2.1 upstream merge. This phase is pure verification and test fixup — the actual code adaptation happens during Phase 2.1 merge conflict resolution. Phase 3 confirms eviction modules import cleanly, unit tests pass, and key classes instantiate without error.

**Scope change from original roadmap:** Phase 3 was originally "re-apply scheduler logic, adapt GPU model runner." The decision to merge `upstream/main` (Phase 2.1) means the adaptation happens during merge conflict resolution. Phase 3 is now verification that the merge preserved working eviction code.

</domain>

<decisions>
## Implementation Decisions

### Upstream Merge Strategy (Phase 2.1 — prerequisite)
- **D-01:** Merge `upstream/main` (latest HEAD) into `upgrade_vllm` branch. The fork shares ancestor commit `963dc0b865` with upstream/main, enabling a proper git merge.
- **D-02:** Work off upstream `main` branch, not release tag branches. The fork was originally based on upstream `main`, and release tag branches have different commit hashes for the same changes.
- **D-03:** Pull in upstream tags after the merge for version reference.
- **D-04:** Phase 2.1 must be inserted into the roadmap before Phase 3 can execute.

### Merge Conflict Resolution Approach (Phase 2.1)
- **D-05:** Conflict resolution IS the adaptation — eviction code in scheduler.py, gpu_model_runner.py, and other files gets adapted to upstream's latest APIs during merge resolution. No separate adaptation step.
- **D-06:** For files already reconciled in Phase 2 (struct fields, IPC layer), use the Phase 2 versions as conflict resolution reference since they already contain upstream-compatible code + eviction additions.
- **D-07:** For files not yet reconciled (scheduler internals, GPU runner internals, serving layer), resolve by taking upstream code and adding eviction code adapted to the current API.

### KV Cache free_blocks
- **D-08:** Adapt eviction's `free_blocks()` call to match upstream's current KV cache coordinator API during conflict resolution. No wrapper or bridge — eviction code speaks upstream's language directly.

### L2 Norm Retrieval
- **D-09:** Adapt L2 norm retrieval in `update_from_output` to match upstream's current return type structure. No type coercion bridges — rewrite to use the upstream structure natively.

### Phase 3 Verification Scope
- **D-10:** Verify all eviction modules import cleanly on the merged codebase.
- **D-11:** Existing unit tests in `tests/eviction/` pass (after test updates).
- **D-12:** Key eviction classes (`EvictionOrchestrator`, `ThoughtSegmenter`, `L2NormCache`, strategy classes) instantiate without error.
- **D-13:** Serving-level smoke test and live server verification stay in Phase 4.

### Source-Inspect Test Handling
- **D-14:** Update source-inspect tests to match the new function signatures and bodies after the merge. Tests should reflect the actual merged code, not the pre-merge v0.14 versions.

### Claude's Discretion
- Exact conflict resolution choices for each file during Phase 2.1 merge
- Order of test updates in Phase 3
- Whether to add any lightweight integration checks beyond imports and unit tests
- Commit granularity within Phase 2.1 and Phase 3

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase 2.1 Merge (prerequisite)
- `upstream/main` — Merge target. Shared ancestor with `upgrade_vllm` at commit `963dc0b865`
- `.planning/phases/02-safe-additions/02-CONTEXT.md` — Phase 2 reconciliation decisions; serves as conflict resolution reference for already-adapted files

### Phase 1 Audit Findings
- `.planning/phases/01-groundwork-audit/01-AUDIT-FINDINGS.md` — v0.19 structural analysis (byte values, field layout, block table API, schedule() structure, client hierarchy)

### Scheduler (merge conflict targets)
- `vllm/v1/core/sched/scheduler.py` — `update_request_mask`, `_process_evictions`, `update_from_output` L2 norm retrieval, `_free_request` cleanup
- `vllm/v1/core/sched/output.py` — `SchedulerOutput.evictable_token_ranges_map`

### GPU Model Runner (merge conflict targets)
- `vllm/v1/worker/gpu_model_runner.py` — `_compute_l2_norms`, block table invalidation, `execute_model` L2 norm call, L2 norm cache init
- `vllm/v1/worker/block_table.py` — `MultiGroupBlockTable` API for block invalidation

### KV Cache Coordinator
- `vllm/v1/core/kv_cache_coordinator.py` — Coordinator hierarchy, `free_blocks()` delegation pattern

### L2 Norm Cache
- `vllm/v1/attention/l2_norm_cache.py` — Singleton cache, differential retrieval, per-request layer filtering

### Eviction Module
- `vllm/thought_eviction/` — All eviction module files (orchestrator, segmenter, strategies, block_utils)

### Tests
- `tests/eviction/` — All existing eviction test files (source-inspect and functional)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- Phase 2 reconciled files (struct fields, IPC layer) already adapted to upstream patterns — serve as conflict resolution templates
- `eviction_changes.patch` — Complete diff of all fork changes; useful reference during merge conflict resolution

### Established Patterns
- `import regex as re` required (not `import re`) — enforced by pre-commit
- msgspec structs use `omit_defaults=True` with `array_like` semantics — field ordering is byte-significant
- Client hierarchy: `EngineCoreClient` ABC → `InprocClient` / `MPClient` → `SyncMPClient` / `AsyncMPClient` → `DPAsyncMPClient` → `DPLBAsyncMPClient`
- KV cache coordinator delegates through `single_type_managers` tuple

### Integration Points
- Merge base: commit `963dc0b865` — ~2527 upstream commits to merge, ~98 fork commits since divergence
- Scheduler `_process_evictions()` calls `kv_cache_manager.free_blocks()` — API may have evolved in upstream
- GPU runner `_compute_l2_norms()` accesses `InputBatch.block_table` — API structure documented in Phase 1 audit
- `update_from_output()` return type changed in upstream — L2 norm retrieval must adapt

</code_context>

<specifics>
## Specific Ideas

- The fork and upstream/main share ancestor `963dc0b865` — this enables a proper `git merge` without `--allow-unrelated-histories`
- Release tag branches (v0.19.0 etc.) have different commit hashes from `main` for the same changes — always work off `upstream/main`
- Phase 2 work is not wasted — reconciled files serve as the "correct answer" for merge conflicts in those files

</specifics>

<deferred>
## Deferred Ideas

- **Roadmap update needed:** Phase 2.1 (upstream merge) must be inserted into ROADMAP.md before planning
- **Requirements update needed:** SCHED-01..06 and GPU-01..05 requirements shift — they're achieved through merge conflict resolution (Phase 2.1), verified in Phase 3
- **PROJECT.md update needed:** "Target: v0.19.0 tag" should become "Target: upstream/main (latest)"

</deferred>

---

*Phase: 03-core-adaptation*
*Context gathered: 2026-04-08*
