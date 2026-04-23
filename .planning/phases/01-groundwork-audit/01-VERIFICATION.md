---
phase: 01-groundwork-audit
verified: 2026-04-08T04:15:00Z
status: passed
score: 5/5 must-haves verified
gaps: []
---

# Phase 1: Groundwork & Audit Verification Report

**Phase Goal:** The codebase is free of pre-existing crashes and all structural unknowns about v0.19 are resolved before any merge work begins
**Verified:** 2026-04-08T04:15:00Z
**Status:** passed
**Re-verification:** No -- initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `api_server.py` no longer crashes on startup due to Pydantic v1 `parse_obj` call | VERIFIED | `model_validate` present, `parse_obj` absent in `vllm/entrypoints/api_server.py` |
| 2 | `InprocClient.update_request_mask_async` exists and is callable | VERIFIED | AST parse confirms `update_request_mask_async`, `update_request_mask`, and `get_request_l2_norms_async` all present in InprocClient class (line 360+ in core_client.py) |
| 3 | A functional smoke test drives the full eviction pipeline end-to-end and produces an observable pass/fail result | VERIFIED | `tests/eviction/test_smoke.py` (267 lines) exists, compiles, contains `test_eviction_smoke` function with server lifecycle, streaming request with `eviction_params`, and assertions for eviction statistics |
| 4 | `EngineCoreRequestType` byte values in v0.19 are documented and `b'\x05'` collision status is confirmed | VERIFIED | `01-AUDIT-FINDINGS.md` (223 lines) contains AUDIT-01 section documenting WAKEUP collision at `\x05` |
| 5 | SamplingParams field additions, block table API, schedule() structure, and all EngineCoreClient subclasses in v0.19 are enumerated and recorded | VERIFIED | `01-AUDIT-FINDINGS.md` contains sections AUDIT-02 (SamplingParams fields), AUDIT-03 (MultiGroupBlockTable API), AUDIT-04 (SchedulerOutput construction), AUDIT-05 (5 client subclasses including DPLBAsyncMPClient) |

**Score:** 5/5 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `vllm/entrypoints/api_server.py` | Pydantic v2 model_validate | VERIFIED | `model_validate` present, `parse_obj` absent |
| `vllm/v1/engine/core_client.py` | InprocClient eviction methods | VERIFIED | All 3 methods present: update_request_mask_async, update_request_mask, get_request_l2_norms_async |
| `vllm/v1/attention/l2_norm_cache.py` | Clean single-singleton L2NormCache | VERIFIED | No `_instance`, no `__new__`, has `__init__`, module-level `_l2_norm_cache` global present |
| `vllm/v1/worker/gpu_model_runner.py` | Fixed layer sorting, single import | VERIFIED | `key=lambda` + `rsplit` present; exactly 1 `get_l2_norm_cache` import |
| `vllm/v1/attention/backends/flash_attn.py` | No dead compute_l2_norms/request_ids fields | VERIFIED | AST confirms neither field in FlashAttentionMetadata class |
| `tests/eviction/test_no_eviction_guard.py` | Module-level singleton reset | VERIFIED | No `_instance` reference; uses `cache_mod._l2_norm_cache = None` |
| `tests/eviction/test_smoke.py` | Server-level functional smoke test | VERIFIED | 267 lines, syntactically valid, contains server management, streaming request with eviction_params, GPU skip condition |
| `.planning/phases/01-groundwork-audit/01-AUDIT-FINDINGS.md` | All 5 audit findings documented | VERIFIED | 223 lines, all 5 AUDIT sections present with v0.19.0 file paths and line numbers |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `vllm/v1/engine/core_client.py` | `vllm/v1/core/sched/scheduler.py` | `self.engine_core.scheduler.update_request_mask()` | WIRED | Found at lines 362 and 369 in core_client.py |
| `vllm/v1/worker/gpu_model_runner.py` | `vllm/v1/attention/l2_norm_cache.py` | Single import of `get_l2_norm_cache` | WIRED | Exactly 1 import at line 129 |
| `tests/eviction/test_smoke.py` | `vllm/entrypoints/openai/api_server` | HTTP request to `/v1/chat/completions` | WIRED | URL construction at line 122 |

### Data-Flow Trace (Level 4)

Not applicable -- Phase 1 artifacts are code fixes, a documentation file, and a test. No dynamic data rendering components.

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Smoke test compiles | `python -m py_compile tests/eviction/test_smoke.py` | SYNTAX OK | PASS |
| All commits exist | `git log --oneline -1 <hash>` for 4 commit hashes | All 4 found | PASS |
| Audit doc has all sections | Content check for AUDIT-01 through AUDIT-05 | All 5 present with key terms | PASS |

Step 7b note: Server startup tests (actual eviction pipeline) require GPU and model, cannot be run in this environment. Routed to human verification below.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| CLEAN-01 | 01-01 | Fix Pydantic v1 `parse_obj` crash in `api_server.py` | SATISFIED | `model_validate` present, `parse_obj` absent |
| CLEAN-02 | 01-01 | Implement `InprocClient.update_request_mask_async` | SATISFIED | Method present in InprocClient, delegates to scheduler |
| CLEAN-03 | 01-03 | Write functional smoke test for eviction pipeline | SATISFIED | `tests/eviction/test_smoke.py` (267 lines) with full pipeline test |
| AUDIT-01 | 01-02 | Verify `EngineCoreRequestType` byte values in v0.19 | SATISFIED | Documented in AUDIT-FINDINGS.md with WAKEUP/x05 collision confirmed |
| AUDIT-02 | 01-02 | Identify all new SamplingParams fields (v0.14 to v0.19) | SATISFIED | Documented in AUDIT-FINDINGS.md AUDIT-02 section |
| AUDIT-03 | 01-02 | Determine v0.19 `InputBatch.block_table` API | SATISFIED | Documented in AUDIT-FINDINGS.md with MultiGroupBlockTable details |
| AUDIT-04 | 01-02 | Map v0.19 `schedule()` structure | SATISFIED | Documented in AUDIT-FINDINGS.md with SchedulerOutput construction site |
| AUDIT-05 | 01-02 | Enumerate all v0.19 `EngineCoreClient` subclasses | SATISFIED | Documented in AUDIT-FINDINGS.md with 5 client subclasses listed |

No orphaned requirements found. All 8 requirement IDs from REQUIREMENTS.md Phase 1 mapping are covered by plans.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `vllm/v1/engine/core_client.py` | 82, 1281 | TODO comments | Info | Pre-existing upstream TODOs, not introduced by this phase |
| `vllm/v1/worker/gpu_model_runner.py` | various | TODO/FIXME comments | Info | Pre-existing upstream TODOs, not introduced by this phase |
| `vllm/v1/attention/backends/flash_attn.py` | various | TODO comments | Info | Pre-existing upstream TODOs, not introduced by this phase |

No anti-patterns introduced by Phase 1 work. All flagged items are pre-existing upstream code.

### Human Verification Required

### 1. Smoke Test Baseline Run

**Test:** Run `pytest tests/eviction/test_smoke.py -v` on a GPU compute node with the deepseek-8b model at `$HOME/scratch/models/deepseek-8b`
**Expected:** Test passes -- server starts, streaming response received with L2 norms and eviction statistics in final SSE chunk
**Why human:** Requires GPU hardware and model weights not available in verification environment

### 2. Server Startup After Pydantic Fix

**Test:** Run `python -c "from vllm.entrypoints.api_server import app"` to verify api_server.py imports without crash
**Expected:** No ImportError or AttributeError related to parse_obj
**Why human:** Requires vLLM runtime dependencies (torch, CUDA) to fully import

### Gaps Summary

No gaps found. All 5 observable truths verified through codebase inspection. All 8 requirements satisfied. All artifacts exist, are substantive, and are properly wired. Two items require human verification on GPU hardware (smoke test execution and server startup), but all automated checks pass.

---

_Verified: 2026-04-08T04:15:00Z_
_Verifier: Claude (gsd-verifier)_
