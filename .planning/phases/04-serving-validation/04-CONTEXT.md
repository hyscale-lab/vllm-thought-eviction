# Phase 4: Serving & Validation - Context

**Gathered:** 2026-04-09
**Status:** Ready for planning

<domain>
## Phase Boundary

Wire the serving layer to activate the eviction orchestrator on requests with `eviction_params`, fix all remaining test failures, and verify the server handles live eviction requests end-to-end. This is the final integration phase — scheduler and GPU runner are fully adapted (Phase 3); Phase 4 connects the API layer to the eviction pipeline and confirms the full system works.

**Pre-done (no work needed):**
- SERVE-01: `EvictionParams` + `eviction_params` field already present in `vllm/entrypoints/openai/chat_completion/protocol.py`
- SERVE-03: `/v1/attention/l2_norms` already implemented in `vllm/entrypoints/openai/extensions/attention_tools.py`; router registered in `vllm/entrypoints/openai/api_server.py` (line 258)

**Remaining work:**
- SERVE-02: Port eviction wiring into v0.19 `serving.py`
- SERVE-04: All eviction tests pass (5 pre-existing failures + test_serving_integration.py adaptation)
- SERVE-05: Server starts and handles live eviction requests
- SERVE-06: Functional smoke test passes end-to-end

</domain>

<decisions>
## Implementation Decisions

### Pre-existing Test Failures
- **D-01:** Fix all 5 pre-existing test failures in Phase 4. These are test-side fixes only — no production code changes.
  - `test_strategies.py` (4 tests): Remove stale `prune_after_tokens` kwargs from `GlobalStrategy.compute_evictable_ranges()` calls — method signature doesn't accept it
  - `test_orchestrator.py` (1 test): `test_run_eviction_cycle_applies_absolute_offset` — fix mock setup so `update_request_mask` is called correctly
  - Target: 109/109 tests passing before Phase 4 closes

### Serving.py Wiring Placement
- **D-02:** Claude's discretion. The planner must read both pre-merge serving.py (`git show 2ec9a65e84b:vllm/entrypoints/openai/chat_completion/serving.py`) and current v0.19 serving.py in full before deciding the insertion point.
  - Pre-merge pattern (reference): orchestrator wraps `result_generator` before it enters the streaming path; `chat_completion_stream_generator` sees the wrapped stream transparently
  - v0.19 context: `create_chat_completion` assembles `result_generator` at line 329, passes it to `chat_completion_stream_generator` (line 332) for stream mode
  - Planner may deviate from pre-merge pattern if v0.19's structure requires it — correctness over fidelity to pre-merge

### Plan Structure
- **D-03:** Two plans.
  - **Plan 04-01:** Serving wiring + all test fixes
    - Port eviction wiring into `serving.py` (SERVE-02)
    - Fix all 5 pre-existing test failures (test-side only)
    - Adapt `test_serving_integration.py` to match v0.19 serving.py structure
    - Verify all 109 eviction tests pass (SERVE-04)
  - **Plan 04-02:** Server startup + smoke test
    - Start vLLM server with DeepSeek-8B model
    - Send live chat completion request with `eviction_params`
    - Verify response contains eviction statistics
    - Smoke test (`tests/eviction/test_smoke.py`) passes end-to-end (SERVE-05, SERVE-06)

### Claude's Discretion
- Exact insertion point for orchestrator in v0.19 `serving.py` (read both pre-merge and v0.19 in full)
- Commit granularity within each plan
- Whether `test_serving_integration.py` tests need structural changes or just source-inspect pattern updates
- Smoke test startup flags and any v0.19-specific server arguments

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Pre-merge Reference (serving wiring source)
- Pre-merge commit `2ec9a65e84b` — Working eviction wiring in serving.py. Retrieve with `git show 2ec9a65e84b:vllm/entrypoints/openai/chat_completion/serving.py`

### Serving Layer (adaptation targets)
- `vllm/entrypoints/openai/chat_completion/serving.py` — v0.19 serving.py; eviction wiring goes here (SERVE-02)
- `vllm/entrypoints/openai/chat_completion/protocol.py` — `EvictionParams`, `ChatCompletionRequest` with `eviction_params` (already wired, verify only)
- `vllm/entrypoints/openai/extensions/attention_tools.py` — `/v1/attention/l2_norms` and `/v1/attention/update_mask` endpoints (already wired, verify only)
- `vllm/entrypoints/openai/extensions/protocol.py` — `UpdateMaskRequest`, `L2NormsRequest`
- `vllm/entrypoints/openai/api_server.py` — Router registration (attention_router at line 258)

### Eviction Module (already present)
- `vllm/thought_eviction/orchestrator.py` — `EvictionOrchestrator`, `wrap_stream()`
- `vllm/thought_eviction/strategies.py` — All strategy classes (`GlobalStrategy`, `ThoughtMinStrategy`, `ThoughtAvgStrategy`, `RandomStrategy`)
- `vllm/thought_eviction/orchestrator.py` — `update_request_mask` (mock target for test fix)

### Tests (all targets for SERVE-04)
- `tests/eviction/test_serving_integration.py` — Source-inspect tests for serving.py eviction wiring (needs adaptation after SERVE-02)
- `tests/eviction/test_protocol_extension.py` — Protocol tests (should pass as-is)
- `tests/eviction/test_strategies.py` — 4 pre-existing failures to fix
- `tests/eviction/test_orchestrator.py` — 1 pre-existing failure to fix
- `tests/eviction/test_smoke.py` — Functional smoke test (SERVE-06)
- `tests/eviction/` — All other eviction tests (104/109 currently passing)

### Phase 3 Context (prior adaptation decisions)
- `.planning/phases/03-core-adaptation/03-CONTEXT.md` — API adaptation decisions, block table, L2 norm retrieval pattern
- `.planning/phases/03-core-adaptation/03-VERIFICATION.md` — Verified state of scheduler and GPU runner
- `.planning/phases/03-core-adaptation/deferred-items.md` — 5 pre-existing failures root cause analysis

### Async LLM (eviction methods)
- `vllm/v1/engine/async_llm.py` — `update_request_mask` (line 1018), `get_request_l2_norms` (line 1037) — already implemented in Phase 2

</canonical_refs>

<code_context>
## Existing Code Insights

### Already Wired (verify-only in Phase 4)
- `protocol.py`: `EvictionParams` at line 153, `eviction_params` field on `ChatCompletionRequest` at line 361
- `attention_tools.py`: `/v1/attention/l2_norms` POST endpoint using `L2NormsRequest`, calls `engine.get_request_l2_norms()`
- `api_server.py` line 258: `from vllm.entrypoints.openai.extensions.attention_tools import router as attention_router` + `app.include_router(attention_router, prefix="")`

### Serving.py v0.19 Structure
- `create_chat_completion` at line 204
- `result_generator` assembled at line 329: `(result_generator,) = generators`
- Stream path at line 331: `if request.stream: return self.chat_completion_stream_generator(request, result_generator, ...)`
- `chat_completion_stream_generator` at line 498 — ~400 lines, handles tool parsing, harmony streaming, reasoning parsers, usage stats, SSE formatting
- Pre-merge pattern: wrap `result_generator` before it enters streaming logic (one injection point in `create_chat_completion`)

### Pre-existing Test Failure Root Causes (from deferred-items.md)
- `test_strategies.py`: Tests call `GlobalStrategy.compute_evictable_ranges(thoughts, l2_norms, prune_after_tokens=N)` but method signature doesn't accept `prune_after_tokens`
- `test_orchestrator.py`: Mock setup for `test_run_eviction_cycle_applies_absolute_offset` never triggers `update_request_mask` call

### Phase 2 IPC Layer
- All client eviction methods fully implemented (Phase 2): `update_request_mask_async`, `get_request_l2_norms_async` on all 5 client subclasses
- `async_llm.py` methods are ready and callable

### Smoke Test Configuration
- Model: `$HOME/scratch/models/deepseek-8b`
- Port: 8192 (non-default to avoid conflicts)
- Server: `python -m vllm.entrypoints.openai.api_server` (v0.19 OpenAI-compatible server)
- GPU requirement: auto-skip on CPU-only nodes (already in test_smoke.py)

</code_context>

<specifics>
## Specific Ideas

- The smoke test sends a math problem to trigger extended `<think>` reasoning from DeepSeek with `strategy: "thought_min"`, `keep_ratio: 0.6`, `prune_after_tokens: 50` — aggressive settings to ensure eviction fires
- SERVE-01 and SERVE-03 are already complete — planner should verify (grep/read) rather than re-implement

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 04-serving-validation*
*Context gathered: 2026-04-09*
