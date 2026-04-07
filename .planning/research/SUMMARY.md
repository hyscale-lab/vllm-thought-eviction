# Research Summary: vLLM v0.14 → v0.19 Upgrade

**Project:** hyscale-lab/vllm-thought-eviction — upgrade fork from v0.14.0rc2 to v0.19.0
**Synthesized:** 2026-04-07

## Executive Summary

Upgrading the thought eviction fork from vLLM v0.14.0rc2 to v0.19.0 is a **pure internal API migration** — not a dependency problem. Python 3.10+, PyTorch 2.9.1, FlashInfer 0.5.3, CUDA 12.x, and the build system are all identical between the two versions. The environment requires zero changes. The challenge is entirely in re-applying 37 modified files across vLLM's most volatile internals: the scheduler, the GPU model runner, the IPC envelope structs, and the KV cache manager — all of which were substantially refactored between v0.14 and v0.19.

The recommended approach is a **manual, file-by-file re-application** of the eviction patch rather than a mechanical `git merge` or `git apply`. Four integration points require structural adaptation — not just re-patching: (1) `update_from_output` now returns a multi-client dict requiring a logic rewrite of the L2 norm retrieval loop; (2) the KV cache manager split into a three-class coordinator hierarchy; (3) the GPU model runner's private block table API (`block_table.block_tables`, `.block_table.np`) was refactored for ubatch support and is near-certain to crash; and (4) new `EngineCoreClient` subclasses in v0.19 must receive the eviction IPC methods.

The most critical risk is the test suite's design: all 9 eviction tests use `inspect.getsource()` rather than functional assertions. A green CI build can coexist with a completely broken runtime. The single most valuable mitigation — before any merge work — is writing a functional smoke test that drives the full eviction pipeline end-to-end.

## Key Findings

**Stack:** No dependency changes required. All upgrade work is Python code merging. The `EngineCoreRequestType.UPDATE_MASK = b'\x05'` enum value may collide with a v0.19 upstream addition (check first, 10-second grep). `array_like=True, omit_defaults=True` msgspec encoding on IPC structs makes field ordering byte-significant.

**Features:** Two msgspec positional encoding hazards dominate (`SamplingParams` and `EngineCoreOutput`). `FlashAttentionMetadata` carries two dead fields (`compute_l2_norms`, `request_ids`) confirmed unused by the kernel — removing them eliminates an entire conflict zone. Pydantic v1 `parse_obj` in `api_server.py` is a pre-existing crash that must be fixed before the merge. `InprocClient` is missing `update_request_mask_async` — two-line fix.

**Architecture:** The scheduler gained a formal `SchedulerInterface` ABC. `SchedulerOutput` gained 6+ new upstream fields. `gpu_model_runner.py` had `InputBatch`/`CachedRequestState` extracted and micro-batching added — every eviction hunk needs manual placement. New `DPAsyncMPClient` and actor-based clients must receive the three eviction IPC methods. Flash attention helpers moved to `fa_utils.py`.

**Pitfalls:** Top 5 by severity: (1) source-inspect tests mask broken runtime, (2) block table private API gone in v0.19, (3) `update_from_output` return type change requires logic rewrite, (4) msgspec positional encoding corruption, (5) `EngineCoreRequestType` byte collision. All have clear prevention steps.

## Critical Risks (Ranked)

| # | Risk | Severity | Prevention |
|---|------|----------|------------|
| 1 | Source-inspect tests mask broken runtime | CRITICAL | Write functional smoke test before merge |
| 2 | Block table private API gone in v0.19 | CRITICAL | Diff `block_table.py` before applying model runner hunks |
| 3 | `update_from_output` return type changed | HIGH | Rewrite L2 norm loop for multi-client dict |
| 4 | msgspec positional encoding corruption | HIGH | Always append eviction fields last in v0.19 structs |
| 5 | `EngineCoreRequestType` byte collision | HIGH | Check v0.19 enum before applying |
| 6 | `SchedulerOutput` silent empty dict | HIGH | Grep all construction sites after applying |
| 7 | `FlashAttentionMetadata.build()` signature change | MEDIUM | Remove dead fields instead of carrying forward |
| 8 | Missing eviction methods on new client subclasses | MEDIUM | Enumerate all v0.19 client subclasses |
| 9 | Pydantic v1 `parse_obj` crash | MEDIUM | Fix pre-merge (one-line change) |
| 10 | Scheduler hunk misplacement | MEDIUM | Use semantic anchors, not line offsets |

## Recommended Approach

**Manual file-by-file re-application** in this order:

1. New files first (zero conflict): `vllm/thought_eviction/`, `l2_norm_cache.py`, `tests/eviction/`
2. Simple field additions: `sampling_params.py`, `outputs.py`, `engine/__init__.py`, `sched/output.py`
3. Interface extensions: `worker_base.py`, `gpu_worker.py`, `core_client.py` (all subclasses)
4. Engine methods: `async_llm.py`, `engine/core.py`
5. Scheduler logic: `scheduler.py` — requires adaptation for new return type
6. Model runner: `gpu_model_runner.py` — requires full manual placement
7. Serving integration: `api_server.py`, `serving.py`, `protocol.py`, `flash_attn.py`

## Suggested Phase Structure

| Phase | Goal | Risk | Key Files |
|-------|------|------|-----------|
| 1. Pre-Upgrade Cleanup | Fix pre-existing bugs, write smoke test | LOW | `api_server.py`, `core_client.py`, new test file |
| 2. Fetch & Audit | Resolve 5 open questions with targeted diffs | LOW | No code changes, just investigation |
| 3. New Files & Simple Additions | Copy eviction module, append fields to structs | LOW | `thought_eviction/`, `sampling_params.py`, `outputs.py`, `__init__.py` |
| 4. Interface & IPC Layer | Extend all client/worker subclasses | MEDIUM | `worker_base.py`, `core_client.py`, `async_llm.py`, `core.py` |
| 5. Scheduler Adaptation | Re-apply scheduler with logic rewrite | HIGH | `scheduler.py`, `output.py`, KV cache files |
| 6. GPU Model Runner | Adapt block table API, re-apply eviction hooks | HIGH | `gpu_model_runner.py`, `flash_attn.py` |
| 7. Serving & Validation | Wire serving layer, run all tests + live server | MEDIUM | `api_server.py`, `serving.py`, `protocol.py` |

## Research Gaps

5 bounded questions resolvable with targeted git commands in Phase 2:
1. What byte values does v0.19 use in `EngineCoreRequestType`?
2. What fields did upstream add to `SamplingParams` between v0.14 and v0.19?
3. What is v0.19's `InputBatch.block_table` public API?
4. What is v0.19's `schedule()` ubatch loop structure?
5. What `EngineCoreClient` subclasses exist in v0.19?

---
*Synthesized from: STACK.md, FEATURES.md, ARCHITECTURE.md, PITFALLS.md*
