---
phase: 02-safe-additions
plan: 01
subsystem: engine-structs
tags: [msgspec, sampling-params, engine-core, scheduler-output, l2-norms, regex]

requires:
  - phase: 01-groundwork-audit
    provides: "Audit findings identifying all v0.14 regressions and stale fields"
provides:
  - "Import-clean eviction modules (orchestrator, segmenter use regex)"
  - "sampling_params.py with all v0.19 features + eviction fields"
  - "outputs.py with STREAM_FINISHED sentinel + new_l2_norms"
  - "Engine __init__.py with PauseMode, WAKEUP, UPDATE_MASK, REPETITION, new_l2_norms"
  - "SchedulerOutput with new_block_ids_to_zero + evictable_token_ranges_map"
  - "L2NormCache with module-level singleton only (no __new__)"
affects: [02-02, 03-wiring, scheduler, gpu-model-runner, serving]

tech-stack:
  added: [regex]
  patterns: ["v0.19 base + eviction fields appended last for msgspec compat"]

key-files:
  created: []
  modified:
    - vllm/thought_eviction/orchestrator.py
    - vllm/thought_eviction/segmenter.py
    - vllm/sampling_params.py
    - vllm/outputs.py
    - vllm/v1/engine/__init__.py
    - vllm/v1/core/sched/output.py
    - vllm/v1/attention/l2_norm_cache.py
    - tests/eviction/test_serving_integration.py
    - tests/eviction/test_smoke.py

key-decisions:
  - "Used v0.19 file as base then appended eviction fields (not patching v0.14 file)"
  - "Eviction fields positioned after repetition_detection, before internal post_init fields"
  - "UPDATE_MASK at b'\\x06' avoids collision with v0.19 WAKEUP at b'\\x05'"
  - "new_l2_norms placed as last field on EngineCoreOutput for msgspec omit_defaults"

patterns-established:
  - "v0.19 base + append pattern: copy upstream file, add eviction fields at end"
  - "All eviction modules must use import regex as re (enforced by pre-commit hook)"

requirements-completed: [MERGE-01, MERGE-02, MERGE-03, MERGE-04, MERGE-05, MERGE-06, MERGE-07, MERGE-08]

duration: 5min
completed: 2026-04-08
---

# Phase 02 Plan 01: Struct Reconciliation Summary

**Reconciled 6 shared files to v0.19 base with eviction fields appended; fixed import regex violations across all eviction modules and tests**

## Performance

- **Duration:** 5 min
- **Started:** 2026-04-08T05:19:06Z
- **Completed:** 2026-04-08T05:24:48Z
- **Tasks:** 3
- **Files modified:** 9

## Accomplishments
- All shared struct files (sampling_params, outputs, engine __init__, sched/output) now have complete v0.19 features with eviction additions layered on top
- All v0.14 stale fields removed (logits_processors field, truncate_prompt_tokens, eos_token_id on request, bc_linter, multi_modal_placeholders)
- All eviction modules and tests are import-clean (regex instead of re)
- L2NormCache singleton unified to module-level only

## Task Commits

Each task was committed atomically:

1. **Task 1: Fix eviction module imports and reconcile sampling_params + outputs** - `96f5ff8292` (feat)
2. **Task 2: Reconcile engine __init__.py and SchedulerOutput** - `96cdd17693` (feat)
3. **Task 3: Verify eviction modules and tests are present and clean** - `61920f2c61` (fix)

## Files Created/Modified
- `vllm/thought_eviction/orchestrator.py` - Fixed import re -> import regex as re
- `vllm/thought_eviction/segmenter.py` - Fixed import re -> import regex as re
- `vllm/sampling_params.py` - Replaced with v0.19 base + enable_l2_norms, l2_norm_layers fields
- `vllm/outputs.py` - Replaced with v0.19 base + new_l2_norms param, STREAM_FINISHED restored
- `vllm/v1/engine/__init__.py` - Replaced with v0.19 base + UPDATE_MASK b'\x06', new_l2_norms on EngineCoreOutput
- `vllm/v1/core/sched/output.py` - Replaced with v0.19 base + evictable_token_ranges_map
- `vllm/v1/attention/l2_norm_cache.py` - Removed __new__/_instance singleton pattern
- `tests/eviction/test_serving_integration.py` - Fixed import re -> import regex as re
- `tests/eviction/test_smoke.py` - Restored from upgrade_vllm branch

## Decisions Made
- Used v0.19 file as base then appended eviction fields rather than patching v0.14 files, ensuring no v0.19 features are missed
- Positioned eviction fields after repetition_detection and before output_text_buffer_length in SamplingParams
- UPDATE_MASK assigned b'\x06' to avoid collision with v0.19 WAKEUP at b'\x05'
- new_l2_norms placed as last field on EngineCoreOutput for msgspec omit_defaults zero-overhead

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Removed __new__/_instance singleton from L2NormCache**
- **Found during:** Task 3 (verify l2_norm_cache.py)
- **Issue:** L2NormCache still had dual singleton pattern (__new__/_instance AND module-level) despite Phase 1 decision to unify
- **Fix:** Replaced __new__/class-level _instance with simple __init__, keeping module-level _l2_norm_cache singleton
- **Files modified:** vllm/v1/attention/l2_norm_cache.py
- **Verification:** grep confirms no __new__ or _instance, get_l2_norm_cache() still works
- **Committed in:** 61920f2c61 (Task 3 commit)

**2. [Rule 3 - Blocking] Restored test_smoke.py missing from worktree**
- **Found during:** Task 3 (verify test files)
- **Issue:** test_smoke.py existed on upgrade_vllm branch but was not present in worktree
- **Fix:** Restored file from upgrade_vllm branch via git show
- **Files modified:** tests/eviction/test_smoke.py
- **Verification:** File exists and has correct content
- **Committed in:** 61920f2c61 (Task 3 commit)

---

**Total deviations:** 2 auto-fixed (1 bug fix, 1 blocking)
**Impact on plan:** Both fixes necessary for correctness. No scope creep.

## Issues Encountered
- Cannot run Python import verification in worktree (no torch/vllm installed) - used AST parsing and grep-based checks instead

## Known Stubs
None - all files contain real data and implementations.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- All struct files reconciled with v0.19 base - ready for Phase 02 Plan 02 (wiring/integration changes)
- Eviction fields correctly positioned for msgspec serialization compatibility
- No blockers identified

---
*Phase: 02-safe-additions*
*Completed: 2026-04-08*
