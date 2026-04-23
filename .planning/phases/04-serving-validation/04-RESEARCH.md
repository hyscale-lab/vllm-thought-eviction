# Phase 4: Serving & Validation - Research

**Researched:** 2026-04-09
**Domain:** vLLM serving layer eviction wiring, test failure root causes, integration test adaptation
**Confidence:** HIGH (all findings from direct source inspection — no external lookups needed)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Fix all 5 pre-existing test failures in Phase 4. These are test-side fixes only — no production code changes.
  - `test_strategies.py` (4 tests): Remove stale `prune_after_tokens` kwargs from `GlobalStrategy.compute_evictable_ranges()` calls
  - `test_orchestrator.py` (1 test): `test_run_eviction_cycle_applies_absolute_offset` — fix mock setup so `update_request_mask` is called correctly
  - Target: 109/109 tests passing before Phase 4 closes
- **D-02:** Planner must read both pre-merge and v0.19 serving.py in full before deciding insertion point. Pre-merge pattern is reference; v0.19 correctness takes priority.
- **D-03:** Two plans: Plan 04-01 (serving wiring + test fixes), Plan 04-02 (server startup + smoke test).

### Claude's Discretion
- Exact insertion point for orchestrator in v0.19 `serving.py`
- Commit granularity within each plan
- Whether `test_serving_integration.py` tests need structural changes or just source-inspect pattern updates
- Smoke test startup flags and any v0.19-specific server arguments

### Deferred Ideas (OUT OF SCOPE)
- None — discussion stayed within phase scope
</user_constraints>

---

## Summary

Phase 4 requires two distinct changes: (1) porting the eviction wiring from the pre-merge `serving.py` reference into the current v0.19 `serving.py`, and (2) fixing 5 pre-existing test failures that are pure test-side API mismatches.

The pre-merge reference (commit `2ec9a65e84b`) contains a fully-working eviction wiring pattern. The current v0.19 `serving.py` has a different structure — notably, `chat_completion_stream_generator` accepts a `reasoning_parser` positional arg instead of an `orchestrator` kwarg, and the non-stream path wraps its generator call in try/except. These structural differences mean the porting is not a mechanical copy-paste; the planner must adapt the block to match v0.19's call signatures.

The test failures are precisely diagnosable:
- Four `test_strategies.py` failures: tests pass `prune_after_tokens=N` as a kwarg to `GlobalStrategy.compute_evictable_ranges()`, which only accepts `(l2_norms, keep_ratio)`. Solution: remove the stale kwarg from test call sites.
- One `test_orchestrator.py` failure: test sets `reasoning_start_token_offset=100` but only provides 32 L2 norms, causing `accumulated_l2_norms[100:]` to be empty, so `GlobalStrategy` produces no ranges and `update_request_mask` is never called. Solution: set offset to 0 (or any value where norms remain after slicing).

`test_serving_integration.py` currently fails entirely because `serving.py` imports trigger `ImportError: cannot import name 'NamedToolChoice' from mistral_common`. This is a transitive import issue. After SERVE-02 adds the eviction wiring, the tests will fail for assertion reasons too (eviction wiring not present). The 6 integration tests are source-inspection tests that check for exact strings in `create_chat_completion`'s source.

**Primary recommendation:** Port pre-merge eviction block verbatim for the sampling_params and orchestrator wiring sections; adapt only the `chat_completion_stream_generator` call to include the `orchestrator` kwarg (v0.19 signature must gain this parameter).

---

## 1. Pre-merge Wiring Pattern (Reference: commit 2ec9a65e84b)

### Import (pre-merge line 86)
```python
from vllm.thought_eviction.orchestrator import EvictionOrchestrator
```
This is the only new top-level import needed.

### Section A: sampling_params enable_l2_norms (pre-merge lines 392-396)
Located inside the `for i, engine_input in enumerate(engine_inputs):` loop, after `sampling_params = request.to_sampling_params(...)` is assigned, before `self._log_inputs(...)`:

```python
# Phase 6: Set per-request L2 norm flag for IPC to EngineCore worker.
# This flag travels with SamplingParams through EngineCoreRequest
# to the GPU worker process, where it gates norm computation.
# Random strategy selects thoughts uniformly — no L2 norms needed.
if (request.eviction_params is not None
        and request.eviction_params.strategy != "random"):
    sampling_params.enable_l2_norms = True
    sampling_params.l2_norm_layers = request.eviction_params.l2_norm_layers
```

### Section B: Non-streaming validation (pre-merge lines 450-456)
Located after `(result_generator,) = generators`, before `if request.stream:`:

```python
# Eviction requires streaming — reject non-streaming requests early.
if request.eviction_params is not None and not request.stream:
    return self.create_error_response(
        "eviction_params requires stream=true. "
        "Server-side eviction operates on the streaming token "
        "pipeline and cannot run on non-streaming requests.",
    )
```

### Section C: Orchestrator instantiation + wrap_stream (pre-merge lines 459-469)
Located inside `if request.stream:` block, before the `return self.chat_completion_stream_generator(...)` call:

```python
# D-01: Wrap stream with eviction orchestrator when eviction_params present
orchestrator = None
if request.eviction_params is not None:
    orchestrator = EvictionOrchestrator(
        eviction_params=request.eviction_params,
        engine_client=self.engine_client,
        tokenizer=tokenizer,
        request_id=request_id,
        block_size=self.engine_client.vllm_config.cache_config.block_size,
    )
    result_generator = orchestrator.wrap_stream(result_generator)
return self.chat_completion_stream_generator(
    request,
    result_generator,
    request_id,
    model_name,
    conversation,
    tokenizer,
    request_metadata,
    orchestrator=orchestrator,   # <-- additional kwarg vs v0.19 current signature
)
```

### Section D: `chat_completion_stream_generator` signature (pre-merge lines 641-650)
```python
async def chat_completion_stream_generator(
    self,
    request: ChatCompletionRequest,
    result_generator: AsyncIterator[RequestOutput],
    request_id: str,
    model_name: str,
    conversation: list[ConversationMessage],
    tokenizer: TokenizerLike | None,
    request_metadata: RequestResponseMetadata,
    orchestrator: "EvictionOrchestrator | None" = None,   # <-- new param
) -> AsyncGenerator[str, None]:
```

### Section E: Eviction stats injection (pre-merge lines 1331-1333)
Located inside the per-choice finish_reason block, after `include_continuous_usage` chunk assembly, before `data = chunk.model_dump_json(...)`:

```python
# Phase 4: inject eviction stats on the finish_reason chunk (D-03)
if orchestrator is not None and finish_reason_sent[i]:
    chunk.eviction = orchestrator.build_eviction_payload()
```

---

## 2. v0.19 Serving.py Structure (Current HEAD)

**File:** `vllm/entrypoints/openai/chat_completion/serving.py` (1811 lines)

### `create_chat_completion` — key landmarks
| Line | Code |
|------|------|
| 204 | `async def create_chat_completion(self, request, raw_request=None)` |
| 217 | `tokenizer = self.renderer.tokenizer` |
| 236–238 | `request_id = f"chatcmpl-{self._base_request_id(raw_request, request.request_id)}"` |
| 254–326 | `for i, engine_input in enumerate(engine_inputs):` loop |
| 279–282 | `sampling_params = request.to_sampling_params(max_tokens, self.default_sampling_params)` |
| 284–289 | `self._log_inputs(sub_request_id, engine_input, ...)` |
| 328–329 | `assert len(generators) == 1` / `(result_generator,) = generators` |
| 331 | `if request.stream:` |
| 332–341 | `return self.chat_completion_stream_generator(request, result_generator, request_id, model_name, conversation, tokenizer, request_metadata, reasoning_parser,)` |
| 343–352 | `return await self.chat_completion_full_generator(...)` (no try/except wrap in v0.19 unlike pre-merge) |

### Key structural differences vs. pre-merge

1. **Sampling params block**: v0.19 calls `request.to_sampling_params(max_tokens, self.default_sampling_params)` (2 args). Pre-merge called it with an additional `self.model_config.logits_processor_pattern` arg and a `validate_logits_processors_parameters()` call after. The enable_l2_norms block must be inserted after line 282, before line 284 (`_log_inputs`).

2. **stream_generator signature**: v0.19 currently ends with `reasoning_parser: ReasoningParser | None = None` (line 507). Pre-merge ends with `orchestrator: "EvictionOrchestrator | None" = None`. v0.19 must gain the `orchestrator` kwarg — added as a new final parameter.

3. **Non-stream path**: v0.19 (lines 343–352) calls `chat_completion_full_generator` directly with `return await`. Pre-merge wraps it in try/except for `GenerationError` and `ValueError`. The non-stream eviction validation block goes between line 329 and line 331 in v0.19 — same as pre-merge.

4. **No `reasoning_parser` in pre-merge call**: Pre-merge passes `orchestrator=orchestrator` as the last kwarg. v0.19 currently passes `reasoning_parser` as the last positional arg. After the patch, v0.19's call needs both: existing positional args + `orchestrator=orchestrator` as a new kwarg.

### Injection point summary for Plan 04-01

| Location | What to add |
|----------|-------------|
| Line 1 (imports, after existing vllm imports) | `from vllm.thought_eviction.orchestrator import EvictionOrchestrator` |
| After line 282 (sampling_params assignment, inside for-loop) | enable_l2_norms block (Section A) |
| Between line 329 and line 331 | Non-streaming validation guard (Section B) |
| Between line 329 and `return self.chat_completion_stream_generator(...)` at line 332 | Orchestrator instantiation + wrap_stream (Section C, adapted) |
| `chat_completion_stream_generator` signature (line 507) | Add `orchestrator: "EvictionOrchestrator | None" = None` after `reasoning_parser` |
| Inside finish_reason block (after line 1200, before line 1203 `data = chunk.model_dump_json(...)`) | Eviction stats injection (Section E) |

---

## 3. Test Failures — Root Cause and Exact Fix

### 3.1 test_strategies.py — 4 failures (GlobalStrategy kwarg)

**Failing tests:**
- `test_global_strategy_evicts_highest_norms` (line 110)
- `test_global_strategy_below_prune_threshold_returns_empty` (line 134)
- `test_global_strategy_keep_all_returns_empty` (line 143)
- `test_global_strategy_ranges_are_reasoning_relative` (line 157)

**Root cause:** Each of the 4 tests calls:
```python
strategy.compute_evictable_ranges(
    l2_norms=norms,
    keep_ratio=...,
    prune_after_tokens=...,   # <-- STALE: not in method signature
)
```

**Actual signature** (`vllm/thought_eviction/strategies.py` lines 68–72):
```python
def compute_evictable_ranges(
    self,
    l2_norms: np.ndarray,
    keep_ratio: float,
) -> list[tuple[int, int]]:
```
`prune_after_tokens` is NOT a parameter. The orchestrator enforces it before calling the method; `GlobalStrategy` does not need it.

**Fix:** Remove `prune_after_tokens=N,` from all 4 call sites in `tests/eviction/test_strategies.py`. No other change needed.

Exact lines to modify:
- Line 113: remove `prune_after_tokens=1,`
- Line 135: remove `prune_after_tokens=10,`
- Line 146: remove `prune_after_tokens=1,`
- Line 160: remove `prune_after_tokens=1,`

### 3.2 test_orchestrator.py — 1 failure (wrong offset in test setup)

**Failing test:** `test_run_eviction_cycle_applies_absolute_offset` (line 394)

**Root cause:** The test sets:
```python
orc.accumulated_l2_norms = [float(i) for i in range(32)]  # 32 norms
orc.reasoning_start_token_offset = 100  # absolute offset
```

Inside `_run_eviction_cycle()` (orchestrator.py line 315):
```python
l2_array = np.array(self.accumulated_l2_norms[self.reasoning_start_token_offset:], ...)
```
`accumulated_l2_norms[100:]` on a 32-element list is an empty array. `GlobalStrategy` with an empty array returns `[]`. No ranges → `update_request_mask` is never called.

**Fix (test-side only):** Change `reasoning_start_token_offset = 100` to `reasoning_start_token_offset = 0` (or any value < 32). The test goal is to verify that absolute offset is applied to ranges — offset of 0 still proves the math works because the assertion checks `start >= 100`... wait, no: the assertion checks `start >= 100`. If offset=0, ranges like `(0, 1)` would fail the assertion `start >= 100`.

**Correct fix:** Keep offset at a small positive value AND provide enough norms for the offset + strategy to produce results. Option: set `reasoning_start_token_offset = 5` and `accumulated_l2_norms = [float(i) for i in range(37)]` (37 norms, so 37 - 5 = 32 available to the strategy). The assertion `start >= 5` would then be valid.

Alternatively (simpler): change the test's assertion to match the actual offset:
```python
orc.reasoning_start_token_offset = 5  # was: 100
orc.accumulated_l2_norms = [float(i) for i in range(37)]  # 37 so 32 remain after offset=5
# ...
for start, end in absolute_ranges:
    assert start >= 5, f"Range start {start} is not offset by 5"  # was: >= 100
```

The test's intent (verify offset is added to ranges) is preserved — the number 100 was arbitrary.

---

## 4. test_serving_integration.py — What Each Test Checks and What Will Break

All 6 integration tests use `inspect.getsource(OpenAIServingChat.create_chat_completion)` or `importlib.import_module(...)`. They currently all fail with `ImportError: cannot import name 'NamedToolChoice' from mistral_common` — a transitive import error when loading `serving.py`.

After SERVE-02 adds eviction wiring, the import error may resolve (if the mistral_common issue is fixed), or the tests will fail for content-assertion reasons. Here is what each test checks and whether it will pass after SERVE-02:

| Test | Source-inspect pattern | Will pass after SERVE-02? |
|------|------------------------|--------------------------|
| `test_orchestrator_import_in_serving` (line 21) | `hasattr(mod, 'EvictionOrchestrator')` | YES — import added |
| `test_orchestrator_activation_code_present` (line 37) | Checks `'request.eviction_params is not None'`, `'EvictionOrchestrator('`, `'wrap_stream'`, `'cache_config.block_size'` in source | YES — all present after port |
| `test_block_size_not_hardcoded` (line 60) | `re.findall(r'block_size\s*=\s*\d+', source)` must be empty | YES — uses `cache_config.block_size`, not a literal |
| `test_request_id_not_double_prefixed` (line 76) | `'request_id=request_id'` in normalized source; no `f"chatcmpl-{request_id}"` near orchestrator | YES — pre-merge pattern uses `request_id=request_id` |
| `test_wrap_stream_wires_result_generator` (line 107) | `'result_generator = orchestrator.wrap_stream(result_generator)'` in source | YES — exact string in pre-merge pattern |
| `test_enable_l2_norms_skipped_for_random_strategy` (line 123) | `'strategy != "random"'` in source AND `'enable_l2_norms = True'` in source | YES — both present in Section A wiring |

**Conclusion:** All 6 `test_serving_integration.py` tests should pass once SERVE-02 is complete, assuming the `mistral_common` import error is either resolved or irrelevant (it may be a pre-existing environment issue unrelated to the eviction wiring). The content-assertion patterns exactly match the pre-merge wiring strings — no test content changes needed.

**Note on `test_orchestrator_activation_code_present` (line 50):** checks for `'cache_config.block_size'` in the source. The pre-merge pattern uses `self.engine_client.vllm_config.cache_config.block_size` — this contains `cache_config.block_size` as a substring, so the assertion passes.

---

## 5. Already-Wired Verification

All three items are confirmed present.

### EvictionParams and eviction_params field
**File:** `vllm/entrypoints/openai/chat_completion/protocol.py`
- `EvictionParams` class: line 153
- `eviction_params: EvictionParams | None = Field(...)`: line 361 on `ChatCompletionRequest`
- Status: CONFIRMED PRESENT. No action needed.

### /v1/attention/l2_norms endpoint
**File:** `vllm/entrypoints/openai/extensions/attention_tools.py`
- `router = APIRouter()`: line 16
- `@router.post("/v1/attention/l2_norms")`: line 18
- Also present: `/v1/attention/update_mask` (line 54), `/v1/kv_cache/evict` (line 75), `/v1/attention/l2_norms/config` (line 101)
- Status: CONFIRMED PRESENT. No action needed.

### api_server.py attention router registration
**File:** `vllm/entrypoints/openai/api_server.py`
- Line 258: `from vllm.entrypoints.openai.extensions.attention_tools import router as attention_router`
- Line 260: `app.include_router(attention_router, prefix="")`
- Status: CONFIRMED PRESENT. No action needed.

---

## 6. Smoke Test — What It Does and GPU Detection

**File:** `tests/eviction/test_smoke.py`

### Auto-skip mechanism (lines 30–33)
```python
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Requires GPU — skipping on non-GPU node",
)
```
Uses `torch.cuda.is_available()`. Skips the entire module on CPU-only nodes.

### Server startup (lines 67–116, `vllm_server` fixture)
- Scope: module-level (started once, shared across all tests in file)
- Model: `$HOME/scratch/models/deepseek-8b` — skips if directory not found
- Command:
```bash
python -m vllm.entrypoints.openai.api_server \
    --model $HOME/scratch/models/deepseek-8b \
    --host 127.0.0.1 \
    --port 8192 \
    --dtype auto \
    --max-model-len 4096 \
    --trust-remote-code
```
- `CUDA_VISIBLE_DEVICES=0` set in environment
- Health poll: `GET /health` every 2s, up to 300s timeout
- Cleanup: `proc.terminate()` + `proc.wait(timeout=30)` in finally

### Request body sent (lines 179–197)
```python
{
    "model": MODEL_PATH,
    "messages": [{"role": "user", "content": "Solve step by step: What is the sum of the first 20 prime numbers?"}],
    "stream": True,
    "max_tokens": 2048,
    "eviction_params": {
        "strategy": "thought_min",
        "keep_ratio": 0.6,
        "prune_after_tokens": 50,
        "trigger_mode": "time",
        "l2_norm_layers": [8, 10],
    },
}
```

### Assertions (lines 201–263)
1. At least one SSE chunk received
2. At least one chunk has `finish_reason` set
3. Final chunk contains `eviction` key
4. `eviction` dict has `summary`, `events`, `masked_tokens` keys
5. `summary` has `total_thoughts`
6. If `events` is non-empty, `masked_tokens > 0`

The test does NOT assert that eviction fired (events may be empty); it asserts the pipeline is wired correctly and the payload structure is valid.

---

## Architecture Patterns

### Wiring Pattern: Wrap Before Stream

The canonical pattern (from pre-merge) is:
```
create_chat_completion
  └─ build sampling_params → set enable_l2_norms flag (if eviction, not random)
  └─ assemble result_generator via engine.generate()
  └─ validate: eviction requires stream=True
  └─ if stream:
       └─ if eviction_params: orchestrator = EvictionOrchestrator(...); result_generator = orchestrator.wrap_stream(result_generator)
       └─ return chat_completion_stream_generator(..., orchestrator=orchestrator)
  └─ else: return chat_completion_full_generator(...)
```

`chat_completion_stream_generator` sees the wrapped generator transparently — it iterates `result_generator` which calls through to `orchestrator.wrap_stream()`. No changes are needed inside the streaming loop except the stats injection at the finish_reason point.

### Stats Injection Location (Section E)

In the current v0.19 `chat_completion_stream_generator`, the finish_reason chunk is assembled at lines ~1186–1203:

```python
finish_reason_sent[i] = True   # line 1183

choice_data = maybe_filter_parallel_tool_calls(choice_data, request)
chunk = ChatCompletionStreamResponse(...)  # line 1186

# handle usage stats if requested & if continuous
if include_continuous_usage:                # line 1194
    ...
    chunk.usage = UsageInfo(...)            # line 1197

data = chunk.model_dump_json(...)           # line 1203 — stats injection goes BEFORE here
yield f"data: {data}\n\n"
```

The eviction stats must be injected between the `include_continuous_usage` block and the `model_dump_json` call. The pre-merge condition `if orchestrator is not None and finish_reason_sent[i]:` is correct — `finish_reason_sent[i]` has just been set to `True` at line 1183, so the condition will fire.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead |
|---------|-------------|-------------|
| Eviction orchestrator per-request | Custom streaming wrapper | `EvictionOrchestrator.wrap_stream()` (already in orchestrator.py) |
| Block-aligned range computation | Custom modulo logic | `align_ranges_to_blocks()` from `block_utils.py` |
| Stats payload structure | Ad-hoc dict building | `orchestrator.build_eviction_payload()` (already in orchestrator.py) |

---

## Common Pitfalls

### Pitfall 1: Inserting enable_l2_norms before BeamSearch branching
**What goes wrong:** `enable_l2_norms` is set on `SamplingParams` but `BeamSearchParams` doesn't have this attribute — attribute error.
**How to avoid:** The enable_l2_norms block must be inside the `else:` branch that handles `SamplingParams` (not `BeamSearchParams`). In v0.19, this is the `else:` at line 278 that sets `sampling_params = request.to_sampling_params(...)`. Add it after that assignment.
**v0.19 line:** After line 282 (`sampling_params = request.to_sampling_params(max_tokens, self.default_sampling_params,)`), still inside the `else:` block, before `self._log_inputs(...)` at line 284.

### Pitfall 2: Passing orchestrator as positional arg instead of kwarg
**What goes wrong:** v0.19's `chat_completion_stream_generator` currently ends with `reasoning_parser: ReasoningParser | None = None`. If `orchestrator` is added after this as a new optional kwarg, the call site at line 332 passes `reasoning_parser` positionally — adding `orchestrator=orchestrator` as an extra kwarg works without changing the existing positional call. But if `orchestrator` were inserted before `reasoning_parser` positionally, the existing call would break.
**How to avoid:** Add `orchestrator` as the LAST parameter of `chat_completion_stream_generator`, after `reasoning_parser`. Pass it as a keyword argument at the call site.

### Pitfall 3: Eviction stats injection fires on wrong chunk
**What goes wrong:** `finish_reason_sent[i]` is set to `True` just before the stats injection. If the condition is `if orchestrator is not None and not finish_reason_sent[i]:` (negated), stats never emit.
**How to avoid:** The condition is `if orchestrator is not None and finish_reason_sent[i]:` — positive check. `finish_reason_sent[i]` is set to True at line 1183, then stats injection fires in the same iteration's chunk.

### Pitfall 4: GlobalStrategy test fix removes wrong kwarg
**What goes wrong:** Tests for `ThoughtMinStrategy` and `ThoughtAvgStrategy` also pass `prune_after_tokens` — but those strategy methods DO accept it. Only `GlobalStrategy` lacks it.
**How to avoid:** Only remove `prune_after_tokens` from the 4 `GlobalStrategy` call sites (lines 113, 135, 146, 160 of `test_strategies.py`). Leave all `ThoughtMinStrategy` / `ThoughtAvgStrategy` / `RandomStrategy` calls unchanged.

### Pitfall 5: test_10 orchestrator fix uses wrong assertion after offset change
**What goes wrong:** Changing `reasoning_start_token_offset = 100` to `reasoning_start_token_offset = 5` but leaving `assert start >= 100` causes the test to fail even when `update_request_mask` is correctly called.
**How to avoid:** Update both the offset value AND the assertion bound together. Use `reasoning_start_token_offset = 5` and `assert start >= 5`.

---

## Environment Availability

Step 2.6: SKIPPED — Phase 4 is code/test changes and serving wiring. The smoke test is GPU-dependent but auto-skips; it is not part of the automated test suite gate.

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (configured in `pyproject.toml`) |
| Config file | `pyproject.toml` `[tool.pytest.ini_options]` |
| Quick run command | `python -m pytest tests/eviction/ -q --tb=short` |
| Full suite command | `python -m pytest tests/eviction/ -v` |

### Phase Requirements → Test Map
| Req | Behavior | Test Type | Command |
|-----|----------|-----------|---------|
| SERVE-02 | Eviction orchestrator activated on eviction_params requests | integration | `pytest tests/eviction/test_serving_integration.py -q` |
| SERVE-04 | All 109 eviction tests pass | unit + integration | `pytest tests/eviction/ -q` |
| SERVE-05 | Server starts with v0.19 | smoke (manual) | `pytest tests/eviction/test_smoke.py -v -s` |
| SERVE-06 | Live eviction request works end-to-end | smoke (GPU only) | `pytest tests/eviction/test_smoke.py -v -s` |

### Current Test Counts
- Passing: 98/109 (37 strategies + orchestrator tests passing; 6 serving_integration failing; 4 strategies failing; 1 orchestrator failing; smoke skipped on CPU)

  Note: actual run shows 11 failing, 37 passing — full suite has 109 tests (11 + 98).

### Wave 0 Gaps
None — all test files already exist. No new test files needed for Phase 4.

---

## Sources

### Primary (HIGH confidence — direct source inspection)
- `git show 2ec9a65e84b:vllm/entrypoints/openai/chat_completion/serving.py` — pre-merge wiring reference (lines 86, 392–396, 450–469, 641–650, 1331–1333)
- `vllm/entrypoints/openai/chat_completion/serving.py` HEAD — v0.19 structure (lines 204, 279, 328–341, 498–508, 1183–1203)
- `vllm/thought_eviction/strategies.py` — `GlobalStrategy.compute_evictable_ranges` signature (lines 68–72)
- `vllm/thought_eviction/orchestrator.py` — `_run_eviction_cycle` body (lines 271–407), specifically the `l2_array = np.array(self.accumulated_l2_norms[self.reasoning_start_token_offset:])` slice at line 315
- `tests/eviction/test_strategies.py` — exact line numbers of failing kwarg calls (113, 135, 146, 160)
- `tests/eviction/test_orchestrator.py` — `test_run_eviction_cycle_applies_absolute_offset` setup (lines 394–428)
- `tests/eviction/test_serving_integration.py` — all 6 test assertions (lines 21–138)
- `tests/eviction/test_smoke.py` — server startup flags, request body, assertions (full file)
- `vllm/entrypoints/openai/chat_completion/protocol.py` — `EvictionParams` at line 153, `eviction_params` field at line 361
- `vllm/entrypoints/openai/extensions/attention_tools.py` — `/v1/attention/l2_norms` at line 18
- `vllm/entrypoints/openai/api_server.py` — attention router at line 258–260

### Secondary (MEDIUM confidence)
- Live pytest run output: 11 failed, 37 passed (confirmed failure list matches CONTEXT.md analysis)
- `.planning/phases/03-core-adaptation/deferred-items.md` — prior root cause analysis of test failures

---

## Metadata

**Confidence breakdown:**
- Pre-merge wiring pattern: HIGH — read exact source from git
- v0.19 structure and injection points: HIGH — read exact source from HEAD
- test_strategies.py fix: HIGH — confirmed exact method signature and exact test line numbers
- test_orchestrator.py fix: HIGH — traced execution path through orchestrator code to confirm empty slice is the failure cause
- test_serving_integration.py assertions: HIGH — read full file, matched each pattern to pre-merge source strings
- Smoke test configuration: HIGH — read full test_smoke.py

**Research date:** 2026-04-09
**Valid until:** Stable (no external dependency on fast-moving libraries; all findings from local source)
