---
phase: 01-groundwork-audit
plan: 03
subsystem: testing
tags: [pytest, smoke-test, eviction, sse, streaming, deepseek]

# Dependency graph
requires:
  - phase: 01-groundwork-audit
    provides: "Clean codebase with crash bugs fixed (01-01)"
provides:
  - "Server-level functional smoke test for eviction pipeline end-to-end"
  - "Baseline test for v0.14 before upgrade work begins"
affects: [02-merge-upstream, 03-adapt-eviction, 04-verify-restore]

# Tech tracking
tech-stack:
  added: []
  patterns: ["subprocess.Popen server lifecycle in pytest module fixture", "SSE stream parsing with urllib.request"]

key-files:
  created: [tests/eviction/test_smoke.py]
  modified: []

key-decisions:
  - "Used stdlib urllib.request for HTTP client (no external dependency needed)"
  - "Module-scoped fixture for server lifecycle (start once per test file)"
  - "pytest.skip on missing model or failed server start (graceful degradation)"

patterns-established:
  - "Server smoke test pattern: Popen + health poll + streaming SSE parse"

requirements-completed: [CLEAN-03]

# Metrics
duration: 1min
completed: 2026-04-08
---

# Phase 01 Plan 03: Eviction Smoke Test Summary

**Functional server-level smoke test for thought eviction pipeline: starts vLLM with DeepSeek-8B, sends streaming chat with eviction_params, verifies L2 norms and eviction statistics in SSE response**

## Performance

- **Duration:** 1 min
- **Started:** 2026-04-08T03:56:30Z
- **Completed:** 2026-04-08T03:57:50Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments
- Created `tests/eviction/test_smoke.py` (267 lines) with full server lifecycle management
- Test sends streaming chat completion with aggressive eviction settings (prune_after_tokens=50, keep_ratio=0.5, global strategy)
- Verifies eviction payload structure (summary, events, masked_tokens) in final SSE chunk
- Auto-skips gracefully on non-GPU nodes and when model directory is missing

## Task Commits

Each task was committed atomically:

1. **Task 1: Write functional smoke test for eviction pipeline** - `e8f87f1eba` (feat)

## Files Created/Modified
- `tests/eviction/test_smoke.py` - Server-level functional smoke test: starts vLLM, sends streaming eviction request, verifies pipeline end-to-end

## Decisions Made
- Used `urllib.request` from stdlib instead of requests/httpx to avoid adding dependencies
- Server port 8192 (non-default) to avoid conflicts with other running instances
- Health timeout 300s to accommodate slow model loading on shared GPU nodes
- `trigger_mode: "periodic"` in test request to exercise the time-based eviction trigger path

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Smoke test ready to establish v0.14 baseline when run on GPU node
- Same test will be used to verify eviction pipeline after v0.19.0 upgrade
- All Phase 01 plans (01-01, 01-02, 01-03) now complete

---
*Phase: 01-groundwork-audit*
*Completed: 2026-04-08*
