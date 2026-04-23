# vLLM Thought Eviction — Upgrade to v0.19.0

## What This Is

A fork of vLLM (hyscale-lab/vllm-thought-eviction) that adds thought eviction with L2 norms to the vLLM inference engine. Currently based on vLLM v0.14.0rc2. The goal of this project is to upgrade the fork to vLLM v0.19.0 (commit 2a69949) while preserving all custom thought eviction functionality.

## Core Value

The upgrade must preserve working thought eviction — merging upstream v0.19.0 changes without breaking the eviction orchestrator, L2 norm pipeline, or strategy system.

## Requirements

### Validated

- ✓ Thought eviction with multiple strategies (global, thought_min, thought_avg, random) — existing
- ✓ L2 norm computation in GPU worker with IPC propagation — existing
- ✓ Per-request EvictionOrchestrator middleware — existing
- ✓ OpenAI-compatible chat completion API with eviction_params extension — existing
- ✓ L2 norm polling endpoint (/v1/attention/l2_norms) — existing
- ✓ Eviction test suite (tests/eviction/) — existing

### Active

- [ ] Upgrade vLLM base from v0.14.0rc2 to v0.19.0 (upstream tag, commit 2a69949)
- [ ] Resolve all merge conflicts between upstream v0.19.0 and thought eviction changes
- [ ] Existing eviction tests pass on upgraded codebase
- [ ] Server starts and handles eviction requests correctly on v0.19.0

### Out of Scope

- Refactoring eviction code to use new v0.19.0 APIs — only adapt enough for compatibility
- Adding new eviction features — this is purely an upgrade
- Fixing pre-existing tech debt (dual singleton, UUID suffix issue) — separate effort
- Upgrading beyond v0.19.0 — one version at a time

## Current State

Phase 3 complete — scheduler and GPU model runner eviction logic fully adapted to v0.19 APIs. Scheduler has eviction state, update_request_mask, _process_evictions, L2 norm differential retrieval. GPU runner has block invalidation via MultiGroupBlockTable, _compute_l2_norms adapted for PerLayerAttnMetadata, KV replacement strategies. 104/109 eviction tests pass (5 pre-existing failures). Ready for Phase 4 (Serving & Validation).

## Context

- **Fork repo:** git@github.com:hyscale-lab/vllm-thought-eviction.git (origin)
- **Upstream:** https://github.com/vllm-project/vllm.git
- **Current base:** v0.14.0rc2.dev146+gab2e867c2
- **Target:** v0.19.0 tag (commit 2a69949)
- **Patch file:** eviction_changes.patch contains all custom changes vs upstream
- **Branch strategy:** Work on existing `upgrade_vllm` branch, then PR to hyscale-lab/vllm-thought-eviction main
- **Key integration points likely to conflict:** sampling_params.py, gpu_model_runner.py, input_processor.py, serving.py, scheduler, engine core output structures
- **vLLM v0.14→v0.19 is a major jump** — the v1 engine, scheduler, and worker APIs likely changed significantly

## Constraints

- **Compatibility**: Eviction code hooks into vLLM internals (scheduler, sampler, worker IPC) — these are the most likely conflict zones
- **Branch**: All work happens on `upgrade_vllm` branch before PR to main
- **Verification**: Tests must pass AND server must start with eviction requests working

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Upgrade on dedicated branch (upgrade_vllm) | Isolates upgrade risk from main | — Pending |
| Merge-based upgrade (not rebase) | Preserves commit history, easier conflict resolution | — Pending |
| Minimal adaptation only | Keep eviction code changes to minimum needed for v0.19.0 compat | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd:transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd:complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-04-08 after Phase 1 completion*
