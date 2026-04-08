# Phase 3: Core Adaptation - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-08 (updated)
**Phase:** 03-core-adaptation
**Areas discussed:** Adaptation approach, API adaptation, Test strategy

---

## Session 1 (pre-Phase 2.1)

Original discussion covered upstream merge strategy, phase restructure, KV cache free_blocks path, L2 norm return type, verification approach, and source-inspect test handling. These decisions led to Phase 2.1 insertion and merge execution.

Key outcomes: Merge upstream/main (not tag), conflict resolution IS adaptation, Phase 2.1 inserted, Phase 3 originally framed as "pure verification."

---

## Session 2 (post-Phase 2.1 — context update)

After Phase 2.1 execution revealed that Pattern B files (scheduler, GPU runner) were accepted upstream wholesale with NO eviction code, Phase 3 was reframed from "verification" to "active implementation."

### Adaptation Approach

| Option | Description | Selected |
|--------|-------------|----------|
| Port from pre-merge | Use pre-merge commit as reference, adapt each piece to v0.19 APIs. Preserves proven logic, just updates API calls. | ✓ |
| Write fresh from spec | Re-implement from architecture docs and eviction module interfaces. Cleaner but risks diverging from proven behavior. | |
| You decide | Claude picks best approach per file based on API change magnitude. | |

**User's choice:** Port from pre-merge (Recommended)
**Notes:** Pre-merge reference is commit 2ec9a65e84b.

---

| Option | Description | Selected |
|--------|-------------|----------|
| Both in Phase 3 | Scheduler + GPU runner together — tightly coupled, natural unit. | ✓ |
| Split: scheduler Phase 3, GPU Phase 3.1 | Scheduler first, GPU runner follow-up. Reduces risk but adds overhead. | |

**User's choice:** Both in Phase 3 (Recommended)
**Notes:** Scheduler sends eviction ranges, GPU runner uses them — natural unit.

---

| Option | Description | Selected |
|--------|-------------|----------|
| Import + unit tests | Verify imports, unit tests pass, key classes instantiate. Matches roadmap success criteria. | ✓ |
| Import + unit + integration | Also add lightweight integration checks with mocks. More confidence, more work. | |
| Minimal — just compiles | Only verify no import errors. Tests in Phase 4. | |

**User's choice:** Import + unit tests (Recommended)

---

### API Adaptation

| Option | Description | Selected |
|--------|-------------|----------|
| Native adaptation | Rewrite eviction code to use v0.19 APIs directly. No shims or wrappers. | ✓ |
| Minimal shims | Add thin wrappers to make old code work with new APIs. Faster but adds indirection. | |
| You decide | Claude picks per-API based on complexity. | |

**User's choice:** Native adaptation (Recommended)

---

| Option | Description | Selected |
|--------|-------------|----------|
| Inside output loop | Retrieve L2 norms from L2NormCache while building each EngineCoreOutput. | ✓ |
| Separate pass after | Build outputs first, then iterate to attach L2 norms. | |

**User's choice:** Inside output loop (Recommended)

---

| Option | Description | Selected |
|--------|-------------|----------|
| Group 0 only | Eviction invalidates blocks in KV cache group 0 only — matches primary attention cache. | ✓ |
| All groups | Invalidate across all KV cache groups. More thorough but untested. | |
| You decide | Claude determines based on block table API. | |

**User's choice:** Group 0 only (Recommended)

---

### Test Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Update to match new code | Rewrite source-inspect test expectations to match v0.19-adapted eviction code. | ✓ |
| Drop source-inspect tests | Remove them — brittle, functional tests provide real coverage. | |
| Keep but mark xfail | Expected failures for now, fix in Phase 4. | |

**User's choice:** Update to match new code (Recommended)

---

| Option | Description | Selected |
|--------|-------------|----------|
| Existing tests only | Update and run existing tests/eviction/ suite. No new mock tests. | ✓ |
| Add mock integration tests | Write new tests that mock scheduler/runner to verify eviction methods. | |
| You decide | Claude determines based on existing test coverage. | |

**User's choice:** Existing tests only (Recommended)

---

## Claude's Discretion

- Exact insertion points for eviction code in scheduler.py and gpu_model_runner.py
- Commit granularity within Phase 3 plans
- Plan count and ordering (scheduler-first vs GPU-first vs parallel)
- Whether to add lightweight integration checks beyond import + unit tests
- Specific adaptations needed for each pre-merge function when porting to v0.19

## Deferred Ideas

None — discussion stayed within phase scope.
