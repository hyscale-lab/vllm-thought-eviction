# Phase 2: Safe Additions - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-08
**Phase:** 02-safe-additions
**Areas discussed:** IPC method depth, Module adaptation, Plan granularity, Test handling

---

## IPC Method Depth

| Option | Description | Selected |
|--------|-------------|----------|
| Full implementation (Recommended) | Write complete serialize/send/dispatch logic for all 5 clients. Methods are callable but nothing calls them yet. | ✓ |
| Stubs with NotImplementedError | Add method signatures only, raise NotImplementedError. Quick but defers real work to Phase 3. | |
| Full for ZMQ, stub for DP | Full for InprocClient/SyncMPClient/AsyncMPClient. Stubs for DP clients since DP eviction is out of scope. | |

**User's choice:** Full implementation (Recommended)
**Notes:** Methods callable end-to-end, Phase 3-4 wires the callers.

---

## Module Adaptation

| Option | Description | Selected |
|--------|-------------|----------|
| Adapt during copy (Recommended) | Fix imports and references as each file is copied. Every copied file import-clean against v0.19 on arrival. | ✓ |
| Copy verbatim, fix later | Copy all files as-is, then do a separate import-fixing pass. Known-broken intermediate state. | |
| Copy + minimal fixup | Copy verbatim but only fix imports that would cause immediate ImportError. Defer deeper API fixes to Phase 3. | |

**User's choice:** Adapt during copy (Recommended)
**Notes:** No known-broken intermediate state. Use Phase 1-cleaned file versions.

---

## Plan Granularity

| Option | Description | Selected |
|--------|-------------|----------|
| Two plans (Recommended) | Plan 1: MERGE-01..08 (copy modules, add fields/types). Plan 2: IPC-01..05 (interface layer). Natural dependency boundary. | ✓ |
| Three plans | Plan 1: file copies. Plan 2: field/type additions. Plan 3: interface layer. More granular but more overhead. | |
| One plan | All 13 requirements in a single plan. Simpler orchestration but large scope. | |

**User's choice:** Two plans (Recommended)
**Notes:** IPC plan depends on types/fields from MERGE plan.

---

## Test Handling

| Option | Description | Selected |
|--------|-------------|----------|
| Copy + adapt imports (Recommended) | Copy all test files with imports adapted to v0.19 paths. Source-inspect tests may fail until Phase 3-4 but infrastructure is present. | ✓ |
| Copy only smoke test | Only copy functional smoke test. Skip source-inspect tests since they'll need rewriting. | |
| Copy all verbatim | Copy as-is with broken imports until later phases. | |

**User's choice:** Copy + adapt imports (Recommended)
**Notes:** All tests present and import-clean, even if some can't pass until target files are modified in later phases.

---

## Claude's Discretion

- Exact import path fixups needed during module adaptation
- Internal ordering of work within each plan
- Commit granularity within plans

## Deferred Ideas

None — discussion stayed within phase scope
