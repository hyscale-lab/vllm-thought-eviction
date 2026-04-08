# Phase 3: Core Adaptation - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-08
**Phase:** 03-core-adaptation
**Areas discussed:** Plan granularity, Verification approach, KV cache free_blocks path, L2 norm return type, Upstream merge strategy

---

## Upstream Merge Strategy (user-initiated)

Before gray area discussion began, the user raised a fundamental question: upstream vLLM commits between v0.14 and v0.19 should be in the git history, not bypassed via manual file re-application.

**Discovery:** The fork shares ancestor commit `963dc0b865` with `upstream/main`, enabling a proper git merge. Release tag branches (v0.19.0) have different commit hashes from `main` for the same changes.

| Option | Description | Selected |
|--------|-------------|----------|
| Merge v0.19.0 tag | Brings release tag history; commits differ from main | |
| Rebase onto v0.19.0 | Clean linear history; rewrites all existing commits | |
| Fresh branch + patch | New branch from v0.19.0, apply patch | |
| Merge upstream/main | Shared ancestor exists; proper merge with full upstream history | ✓ |

**User's choice:** Merge latest `upstream/main` HEAD into `upgrade_vllm`
**Notes:** The fork was originally based on upstream `main` (not a release tag), so commits match. Tags pulled in after merge for reference.

---

## Phase Restructure

| Option | Description | Selected |
|--------|-------------|----------|
| Redefine Phase 3 | Phase 3 becomes merge + adapt + verify | |
| Insert Phase 2.1 | New phase for merge; Phase 3 stays as verification | ✓ |

**User's choice:** Insert Phase 2.1 for the merge, Phase 3 becomes pure verification

| Option | Description | Selected |
|--------|-------------|----------|
| Merge-first, adapt after | Separate merge and adaptation steps | |
| Merge includes adaptation | Conflict resolution IS the adaptation | ✓ |
| Keep phases separate | Clean merge, then separate adaptation | |

**User's choice:** Merge includes adaptation — eviction code adapted during conflict resolution

---

## Merge Target

| Option | Description | Selected |
|--------|-------------|----------|
| Latest upstream/main | Most current code, fewer future conflicts | ✓ |
| Find v0.19 point on main | Match original v0.19 upgrade goal | |
| You decide | Claude picks | |

**User's choice:** Latest upstream/main HEAD

---

## KV Cache free_blocks Path

| Option | Description | Selected |
|--------|-------------|----------|
| Adapt to upstream API | Rewrite free_blocks call to match current API | ✓ |
| Minimal bridge | Keep existing call, add adapter wrapper | |
| You decide | Claude picks based on upstream changes | |

**User's choice:** Adapt to upstream API — no wrappers

---

## L2 Norm Return Type

| Option | Description | Selected |
|--------|-------------|----------|
| Adapt to upstream structure | Rewrite L2 norm retrieval to match upstream | ✓ |
| Minimal adaptation | Keep existing logic, add type coercion | |
| You decide | Claude picks based on changes | |

**User's choice:** Adapt to upstream structure — no bridges

---

## Verification Approach

| Option | Description | Selected |
|--------|-------------|----------|
| Import + unit tests | Verify imports, run unit tests, check instantiation | ✓ |
| Full integration test | Write test exercising scheduler/GPU without serving | |
| Skip Phase 3 entirely | Go straight to Phase 4 | |

**User's choice:** Import + unit tests; smoke test stays in Phase 4

---

## Source-Inspect Test Handling

| Option | Description | Selected |
|--------|-------------|----------|
| Update to match merged code | Fix tests to reflect new function bodies | ✓ |
| Skip source-inspect tests | Mark as expected failures | |
| You decide | Claude decides based on change magnitude | |

**User's choice:** Update to match merged code

---

## Plan Granularity

Not discussed separately — superseded by the phase restructure. Phase 2.1 (merge) and Phase 3 (verification) are each a single coherent unit.

---

## Claude's Discretion

- Exact conflict resolution choices per file during Phase 2.1
- Order of test updates in Phase 3
- Commit granularity within phases

## Deferred Ideas

- Roadmap update: Insert Phase 2.1, redefine Phase 3
- Requirements update: SCHED/GPU requirements shift to Phase 2.1 merge resolution
- PROJECT.md update: Target changes from v0.19.0 tag to upstream/main
