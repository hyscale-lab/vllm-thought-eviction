---
phase: 01-groundwork-audit
plan: 01
subsystem: engine
tags: [pydantic, singleton, l2-norm, flash-attention, cleanup]

# Dependency graph
requires: []
provides:
  - Clean baseline codebase with crash bugs fixed (CLEAN-01, CLEAN-02)
  - Unified singleton pattern for L2NormCache (module-level only)
  - Numeric layer sorting in _compute_l2_norms
  - Dead code removed from FlashAttentionMetadata and L2NormCache
affects: [01-02, 01-03, 02-merge]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Module-level singleton for L2NormCache (no __new__ pattern)
    - Pydantic v2 model_validate over parse_obj

key-files:
  created: []
  modified:
    - vllm/entrypoints/api_server.py
    - vllm/v1/engine/core_client.py
    - vllm/v1/attention/l2_norm_cache.py
    - vllm/v1/worker/gpu_model_runner.py
    - vllm/v1/attention/backends/flash_attn.py
    - tests/eviction/test_no_eviction_guard.py

key-decisions:
  - "Kept extra_attn_metadata_args dict structure even after removing L2-specific fields, as other code paths may use it"
  - "Used lazy import for get_l2_norm_cache in InprocClient.get_request_l2_norms_async to avoid circular imports"

patterns-established:
  - "L2NormCache singleton: use module-level _l2_norm_cache + get_l2_norm_cache() only"

requirements-completed: [CLEAN-01, CLEAN-02]

# Metrics
duration: 3min
completed: 2026-04-08
---

# Phase 01 Plan 01: Fix Crash Bugs and Clean Dead Code Summary

**Fixed Pydantic v1 crash in api_server, added InprocClient eviction methods, removed dead FlashAttentionMetadata fields, unified L2NormCache singleton, fixed numeric layer sorting**

## Performance

- **Duration:** 3 min
- **Started:** 2026-04-08T03:48:26Z
- **Completed:** 2026-04-08T03:51:54Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments
- Fixed CLEAN-01: Replaced Pydantic v1 `parse_obj` with v2 `model_validate` in api_server.py to prevent startup crash
- Fixed CLEAN-02: Added `update_request_mask_async`, `update_request_mask`, and `get_request_l2_norms_async` to InprocClient
- Cleaned up D-11 through D-15: removed duplicate import, dead FlashAttentionMetadata fields, dead update_norms method, fixed numeric layer sorting, unified singleton to module-level only

## Task Commits

Each task was committed atomically:

1. **Task 1: Fix CLEAN-01 and CLEAN-02 crash bugs** - `b2da09d437` (fix)
2. **Task 2: Clean up dead code and fix latent bugs D-11 through D-15** - `2b513202cd` (fix)

## Files Created/Modified
- `vllm/entrypoints/api_server.py` - Replaced parse_obj with model_validate
- `vllm/v1/engine/core_client.py` - Added three eviction delegation methods to InprocClient
- `vllm/v1/attention/l2_norm_cache.py` - Removed dual singleton (__new__/_instance), deleted dead update_norms method, kept module-level singleton
- `vllm/v1/worker/gpu_model_runner.py` - Removed duplicate import, removed dead extra_attn_metadata_args for L2 norms, fixed layer sorting to numeric
- `vllm/v1/attention/backends/flash_attn.py` - Removed dead compute_l2_norms/request_ids fields from class and build method
- `tests/eviction/test_no_eviction_guard.py` - Updated _reset_cache to use module-level singleton reset

## Decisions Made
- Kept `extra_attn_metadata_args` dict structure intact even after removing L2-specific fields, since other code paths (e.g., GDN attention) use it
- Used lazy import for `get_l2_norm_cache` inside `InprocClient.get_request_l2_norms_async` to avoid potential circular import issues

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Clean baseline established for merge work in Phase 2
- All crash bugs fixed, dead code removed
- Singleton pattern unified, reducing confusion during v0.19.0 adaptation

---
*Phase: 01-groundwork-audit*
*Completed: 2026-04-08*
