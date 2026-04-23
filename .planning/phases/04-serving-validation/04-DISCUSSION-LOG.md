# Phase 4: Serving & Validation — Discussion Log

**Date:** 2026-04-09
**Mode:** discuss (interactive)

---

## Area 1: Pre-existing Test Failures

**Question:** Fix all 5 pre-existing test failures in Phase 4, or leave deferred?

**Options presented:**
1. Fix them in Phase 4 — ~15 min of test-side edits, gets to 109/109 passing. SERVE-04 reads "all existing eviction tests pass."
2. Leave deferred — they predate this upgrade work, documenting them is sufficient.
3. Fix only if trivial — fix the strategies.py ones (mechanical), leave orchestrator mock one.

**User selected:** Option 1 — Fix them in Phase 4

**Decision captured:** D-01 — Fix all 5 pre-existing test failures (test-side only, no production code changes)

---

## Area 2: Serving.py Wiring Placement

**Question:** Where should `orchestrator.wrap_stream()` be inserted in v0.19's `create_chat_completion`?

**Options presented:**
1. Wrap `result_generator` BEFORE `chat_completion_stream_generator` (pre-merge pattern) — minimal change, one injection point
2. Wrap the SSE output of `chat_completion_stream_generator` — more invasive, orchestrator would need to parse SSE strings
3. Claude's discretion — planner reads both pre-merge and v0.19 serving.py in full and picks cleanest insertion point

**User selected:** Option 3 — Claude's discretion

**Decision captured:** D-02 — Planner reads both `2ec9a65e84b` and v0.19 serving.py before deciding; pre-merge pattern is the reference

---

## Area 3: Plan Structure

**Question:** How many plans for Phase 4?

**Options presented:**
1. Two plans — 04-01 (serving wiring + test fixes), 04-02 (server startup + smoke test); natural code-edit / live-server split
2. One plan — all at once; simpler tracking but mixes code edits with live server validation
3. Three plans — fine-grained but serves 1+2 are small enough to merge

**User selected:** Option 1 — Two plans

**Decision captured:** D-03 — Two plans with natural split at live server boundary

---

*Generated: 2026-04-09*
