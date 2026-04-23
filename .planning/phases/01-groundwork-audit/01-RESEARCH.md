# Phase 1: Groundwork & Audit - Research

**Researched:** 2026-04-08
**Domain:** Pre-upgrade cleanup, codebase audit, smoke test infrastructure
**Confidence:** HIGH

## Summary

Phase 1 addresses three categories of work: (1) fixing pre-existing bugs that would crash the server or block eviction, (2) writing a server-level smoke test that exercises the full eviction pipeline, and (3) auditing v0.19.0 internals to document structural differences before any merge work begins.

The bug fixes are straightforward -- a one-line Pydantic API replacement and a missing method implementation. The smoke test requires a running vLLM server with GPU and the deepseek-8b model, which is available on disk. The audit work is the most critical deliverable: v0.19.0 introduces a confirmed `b'\x05'` byte collision between `WAKEUP` and the fork's `UPDATE_MASK`, completely restructures `SamplingParams` with new fields, replaces the old block table internals with `MultiGroupBlockTable`, and adds two new `EngineCoreClient` subclasses (`DPLBAsyncMPClient`, plus significant changes to existing classes).

**Primary recommendation:** Fix bugs first (they are small and unblock the smoke test), then write the smoke test to establish a passing baseline on v0.14, then perform the audit against the real v0.19.0 tag (commit 2a69949).

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- D-01: Smoke test is a server integration test -- start actual vLLM server, send chat completion with eviction_params, verify full pipeline.
- D-02: Use model at `$HOME/scratch/models/deepseek-8b` for inference.
- D-03: Passing criteria: L2 norms returned, at least one eviction event, final SSE chunk has eviction stats.
- D-04: Run smoke test on current v0.14 codebase first to establish baseline.
- D-05: Audit findings go into `.planning/phases/01-groundwork-audit/01-AUDIT-FINDINGS.md`.
- D-06: Findings use factual summary + code references format. No recommendations.
- D-07: Access v0.19 via git remote/fetch of v0.19.0 tag (commit 2a69949).
- D-08: Bug fixes must be strict minimal fixes.
- D-09: Bug fixes verified via smoke test.
- D-10: `VLLM_KV_REPLACEMENT_STRATEGY` stays as None.
- D-11: Remove duplicate `get_l2_norm_cache` import (gpu_model_runner.py line 171).
- D-12: Remove dead `compute_l2_norms` and `request_ids` fields from FlashAttentionMetadata.
- D-13: Fix `_compute_l2_norms` layer sorting -- use numeric sort.
- D-14: Delete dead `update_norms()` method (l2_norm_cache.py lines 216-238).
- D-15: Fix dual singleton -- keep module-level global only, remove `__new__`-based `_instance`.

### Claude's Discretion
- Smoke test file location and naming within `tests/eviction/`
- Specific prompt text for the smoke test
- Audit doc internal structure
- Order of operations

### Deferred Ideas (OUT OF SCOPE)
- UUID suffix restoration (DEBT-02)
- Memory pre-allocation optimization
- `update_request_mask` overwrite bug
- L2NormCache memory leak in multi-process mode
- Non-streaming eviction path
- Security: no auth on eviction endpoints
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| CLEAN-01 | Fix Pydantic v1 `parse_obj` crash in `api_server.py` | Line 68: replace `UpdateMaskRequest.parse_obj(json_request)` with `UpdateMaskRequest.model_validate(json_request)`. Pydantic 2.12.5 confirmed installed. |
| CLEAN-02 | Implement `InprocClient.update_request_mask_async` | `InprocClient` (core_client.py:273) has no override. Base class raises `NotImplementedError` at line 187-190. Must delegate to `self.engine_core.scheduler.update_request_mask()`. |
| CLEAN-03 | Write functional smoke test for full eviction pipeline | Model at `$HOME/scratch/models/deepseek-8b` exists. No GPU on login node; test must be designed to run on GPU nodes. Existing test patterns in `tests/eviction/` documented. |
| AUDIT-01 | Verify EngineCoreRequestType byte values -- confirm b'\x05' collision | CONFIRMED COLLISION: v0.19.0 has `WAKEUP = b"\x05"`. Fork has `UPDATE_MASK = b'\x05'`. Must use `b'\x06'` or higher. |
| AUDIT-02 | Identify new SamplingParams fields between v0.14 and v0.19 | New fields documented below. Key additions: `_eos_token_id`, `logit_bias`, `allowed_token_ids`, `extra_args`, `bad_words`, `_bad_words_token_ids`, `skip_reading_prefix_cache`, `thinking_token_budget`, `repetition_detection`. Removed: `logits_processors`, `truncate_prompt_tokens`. |
| AUDIT-03 | Determine v0.19 InputBatch.block_table API | v0.19 uses `MultiGroupBlockTable` wrapping multiple `BlockTable` objects. `BlockTable` has `.block_table.np` (via CpuGpuBuffer), `.get_numpy_array()`, `.get_device_tensor()`. Old fork directly accesses `.block_tables` list -- this is gone. |
| AUDIT-04 | Map v0.19 schedule() structure | Single method, no ubatch loop in schedule() itself. Two main loops (RUNNING then WAITING). SchedulerOutput constructed at line 916 with 12 fields. `_update_after_schedule` runs after. Eviction hooks insert before SchedulerOutput construction. |
| AUDIT-05 | Enumerate all v0.19 EngineCoreClient subclasses | 7 classes: `EngineCoreClient` (ABC), `InprocClient`, `MPClient`, `SyncMPClient`, `AsyncMPClient`, `DPAsyncMPClient`, `DPLBAsyncMPClient`. |
</phase_requirements>

## Project Constraints (from CLAUDE.md)

- `import regex as re` required instead of `import re` (enforced by pre-commit hook)
- Direct `import triton` is forbidden
- `pickle`/`cloudpickle` imports blocked
- Ruff linting and formatting enforced (config in `pyproject.toml`)
- `snake_case` for all functions/methods, `PascalCase` for classes
- `logger.error(..., exc_info=True)` for caught exceptions
- `asyncio.CancelledError` must always be re-raised
- `ValueError` for unknown strategy names
- All work on `upgrade_vllm` branch
- GSD workflow enforcement: use `/gsd:` commands for all changes

## Architecture Patterns

### Bug Fix Locations

**CLEAN-01: Pydantic parse_obj crash**
- File: `vllm/entrypoints/api_server.py` line 68
- Current: `mask_request = UpdateMaskRequest.parse_obj(json_request)`
- Fix: `mask_request = UpdateMaskRequest.model_validate(json_request)`
- Context: `UpdateMaskRequest` inherits from `pydantic.BaseModel` (line 39). Pydantic v2 removed `parse_obj()` in favor of `model_validate()`.

**CLEAN-02: InprocClient missing method**
- File: `vllm/v1/engine/core_client.py` lines 273-358
- `InprocClient` has direct access to `self.engine_core` which contains the scheduler
- Must implement both `update_request_mask_async` and `update_request_mask` (sync)
- Pattern: follow existing InprocClient method delegation, e.g. `self.engine_core.scheduler.update_request_mask(request_id, evictable_token_ranges)`
- Also need `get_request_l2_norms_async` -- base class at line 192-195 also raises `NotImplementedError`

**Cleanup items (D-11 through D-15):**
- D-11: `gpu_model_runner.py` line 171 -- duplicate import of `get_l2_norm_cache` (first at line 129)
- D-12: `flash_attn.py` lines 220-221 (`compute_l2_norms`, `request_ids` fields), lines 327-328 (build params), lines 506-507 (build assignment). Also remove from `extra_attn_metadata_args` in gpu_model_runner
- D-13: `gpu_model_runner.py` line 1230 -- change `sorted(attn_metadata_dict.keys())` to numeric sort like `sorted(attn_metadata_dict.keys(), key=lambda s: int(s.split('.')[-1]))`
- D-14: `l2_norm_cache.py` lines 216-238 -- delete entire `update_norms()` method
- D-15: `l2_norm_cache.py` lines 86-95 -- remove `_instance`, `_lock`, and `__new__` override. Keep module-level `_l2_norm_cache` global (lines 346-355)

### Smoke Test Architecture

The smoke test must:
1. Start a vLLM server process with `--model $HOME/scratch/models/deepseek-8b`
2. Send a streaming chat completion request with `eviction_params`
3. Collect SSE chunks, verify:
   - L2 norms present in response data
   - At least one eviction event
   - Final chunk contains eviction statistics
4. Shut down server

**Key patterns from existing tests:**
- Async testing uses `asyncio.run()` wrapping (no pytest-asyncio)
- Factory helpers defined at module top
- Module docstrings list coverage spec codes
- Section separators use 75-char dashed comment blocks

**Server startup for smoke test:**
```python
# Server must be started as a subprocess
# Use vllm.entrypoints.openai.api_server as the entrypoint
# Wait for health check endpoint before sending requests
```

**Prompt requirements:** Must trigger `<think>` block generation from DeepSeek model. A reasoning-heavy prompt (math, logic) will reliably trigger extended thinking.

### Audit Output Structure

Per D-05 and D-06, audit findings go into `01-AUDIT-FINDINGS.md` with:
- Factual summary per finding
- Exact file paths and line numbers in v0.19.0
- Implications for eviction code
- No recommendations

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Server process management in smoke test | Custom subprocess management | `subprocess.Popen` with health check polling | Reliable cleanup, timeout handling |
| SSE parsing in smoke test | Manual SSE chunk parsing | `httpx` with `stream=True` or raw `requests` with line parsing | SSE format is standard, edge cases exist |
| v0.19 source access | Checking out v0.19 files to disk | `git show v0.19.0:path/to/file` | Non-destructive, no working tree changes |

## Common Pitfalls

### Pitfall 1: FETCH_HEAD vs Tag Reference
**What goes wrong:** `git fetch upstream v0.19.0 --depth=1` stores the result in FETCH_HEAD, but FETCH_HEAD can be overwritten by subsequent fetches. The fork's own branch HEAD may be confused with upstream.
**Why it happens:** Shallow fetches with `--depth=1` do not always resolve tags correctly, especially when the repo already has local branches with eviction code.
**How to avoid:** Always use `git show v0.19.0:path` (the tag ref, not FETCH_HEAD) after running `git fetch upstream refs/tags/v0.19.0:refs/tags/v0.19.0`.
**Warning signs:** Seeing eviction code in "upstream" files means you are looking at the fork, not real upstream.

### Pitfall 2: msgspec array_like Field Ordering
**What goes wrong:** `SamplingParams` uses `msgspec.Struct` with `array_like=True` and `omit_defaults=True`. This means fields are serialized positionally (like a tuple), and new fields must go at the end or they break deserialization of existing data.
**Why it happens:** The `enable_l2_norms` and `l2_norm_layers` fields in the fork's v0.14 are appended after the upstream fields. In v0.19, new upstream fields have been added, changing the positional layout.
**How to avoid:** In the upgrade, eviction fields MUST be appended after ALL v0.19 upstream fields, not at the v0.14 positions.
**Warning signs:** Deserialization errors, wrong field values, silent data corruption in IPC.

### Pitfall 3: No GPU on Login Node
**What goes wrong:** Running `pytest tests/eviction/test_smoke.py` on the login node fails because CUDA is not available.
**Why it happens:** The development machine has no GPU (`nvidia-smi` not found, `torch.cuda.is_available()` returns False).
**How to avoid:** The smoke test must be designed to run on GPU compute nodes. Include clear instructions or a wrapper script. Mark with `@pytest.mark.slow_test` or a custom GPU marker.
**Warning signs:** `RuntimeError: No CUDA GPUs are available` on test execution.

### Pitfall 4: InprocClient Method Signature Mismatch
**What goes wrong:** Implementing `update_request_mask_async` on `InprocClient` but the underlying `EngineCore` or `Scheduler` method has a different signature or access pattern.
**Why it happens:** `InprocClient` directly wraps `EngineCore`, but `update_request_mask` is on the `Scheduler`, not on `EngineCore` itself.
**How to avoid:** Check the call chain: `InprocClient` -> `self.engine_core` -> `self.engine_core.scheduler` -> `scheduler.update_request_mask()`. The EngineCore may also have a wrapper method.
**Warning signs:** `AttributeError: 'EngineCore' object has no attribute 'update_request_mask'`.

### Pitfall 5: Dual Singleton Cleanup Order
**What goes wrong:** Removing the `__new__`-based singleton but not updating test code that resets `L2NormCache._instance = None`.
**Why it happens:** Tests in `tests/eviction/` may rely on `_instance` for isolation.
**How to avoid:** Search all test files for `_instance` references before removing the class-level singleton. Provide a `reset()` or `clear()` method on the module-level cache instead.
**Warning signs:** Test failures after D-15 cleanup.

## Code Examples

### CLEAN-01 Fix
```python
# File: vllm/entrypoints/api_server.py, line 68
# Before:
mask_request = UpdateMaskRequest.parse_obj(json_request)
# After:
mask_request = UpdateMaskRequest.model_validate(json_request)
```

### CLEAN-02 Fix Pattern
```python
# File: vllm/v1/engine/core_client.py, inside InprocClient class
async def update_request_mask_async(
    self, request_id: str, evictable_token_ranges: list[tuple[int, int]]
):
    self.engine_core.scheduler.update_request_mask(
        request_id, evictable_token_ranges
    )

def update_request_mask(
    self, request_id: str, evictable_token_ranges: list[tuple[int, int]]
):
    self.engine_core.scheduler.update_request_mask(
        request_id, evictable_token_ranges
    )
```

### D-13 Layer Sorting Fix
```python
# File: vllm/v1/worker/gpu_model_runner.py, line 1230
# Before:
idx_to_name = sorted(attn_metadata_dict.keys())
# After:
idx_to_name = sorted(attn_metadata_dict.keys(),
                     key=lambda name: int(name.rsplit('.', 1)[-1]))
```

### D-15 Singleton Cleanup
```python
# File: vllm/v1/attention/l2_norm_cache.py
# Remove lines 86-95 (__new__, _instance, _lock)
# Change __init__ to _init_cache pattern:

class L2NormCache:
    """..."""
    
    def __init__(self):
        self._request_data: Dict[str, RequestL2NormData] = {}
        self._data_lock = threading.Lock()
        self._enabled = True
        self._l2_norm_layers: Optional[List[int]] = None
        self._skip_layers: Optional[List[int]] = None
        self._request_layer_prefs: Dict[str, Optional[List[int]]] = {}

# Keep module-level singleton (lines 346-355):
_l2_norm_cache: Optional[L2NormCache] = None

def get_l2_norm_cache() -> L2NormCache:
    global _l2_norm_cache
    if _l2_norm_cache is None:
        _l2_norm_cache = L2NormCache()
    return _l2_norm_cache
```

## AUDIT Pre-Research Findings

These are the raw facts the audit tasks will formalize into `01-AUDIT-FINDINGS.md`.

### AUDIT-01: EngineCoreRequestType Byte Values (v0.19.0)
| Value | Name | Notes |
|-------|------|-------|
| `b"\x00"` | ADD | Same as v0.14 |
| `b"\x01"` | ABORT | Same as v0.14 |
| `b"\x02"` | START_DP_WAVE | New in v0.19 (was PROFILE in v0.14) |
| `b"\x03"` | UTILITY | New in v0.19 (replaces several individual types) |
| `b"\x04"` | EXECUTOR_FAILED | New sentinel |
| `b"\x05"` | WAKEUP | **COLLISION with fork's UPDATE_MASK** |

**Confirmed collision:** `UPDATE_MASK = b'\x05'` in the fork collides with `WAKEUP = b'\x05'` in v0.19.0. UPDATE_MASK must use `b'\x06'` or higher.

### AUDIT-02: SamplingParams Field Differences (v0.14 -> v0.19)

**Fields added in v0.19:**
- `_eos_token_id: int | None = None` (new private field)
- `logit_bias: dict[int, float] | None = None`
- `allowed_token_ids: list[int] | None = None`
- `extra_args: dict[str, Any] | None = None`
- `bad_words: list[str] | None = None`
- `_bad_words_token_ids: list[list[int]] | None = None`
- `skip_reading_prefix_cache: bool | None = None`
- `thinking_token_budget: int | None = None`
- `repetition_detection: RepetitionDetectionParams | None = None`

**Fields removed from v0.19 (present in v0.14):**
- `logits_processors: Any | None = None` (was at line 210 in v0.14)
- `truncate_prompt_tokens: Annotated[int, msgspec.Meta(ge=-1)] | None = None` (was at line 215 in v0.14)

**Fields changed:**
- `StructuredOutputsParams` gained `structural_tag` field
- `StructuredOutputsParams` removed `disable_fallback` field
- `SamplingParams` now inherits from `PydanticMsgspecMixin` in addition to `msgspec.Struct`

**Eviction field impact:** `enable_l2_norms` and `l2_norm_layers` (fork additions) must be re-appended AFTER all v0.19 fields (after `repetition_detection`).

### AUDIT-03: Block Table API (v0.19)
- `InputBatch.block_table` is now `MultiGroupBlockTable` (was single `BlockTable` in older versions)
- `MultiGroupBlockTable` wraps multiple `BlockTable` instances, one per KV cache group
- Individual `BlockTable` accessed via `MultiGroupBlockTable[idx]`
- `BlockTable.block_table.np` still exists (via `CpuGpuBuffer`)
- `BlockTable.get_numpy_array()` returns `self.block_table.np`
- `BlockTable.get_device_tensor(num_reqs)` returns GPU tensor
- New: `BlockTable` supports hybrid blocks (kernel_block_size != allocation block_size)
- The fork's direct access to `.block_tables` (plural, a list) is incompatible

### AUDIT-04: schedule() Structure (v0.19)
- Single `schedule()` method at line 348, returns `SchedulerOutput`
- Two main loops: RUNNING requests (line 388-530), WAITING requests (line 556-852)
- `SchedulerOutput` constructed at line 916 with fields (no `evictable_token_ranges_map`)
- `_update_after_schedule()` called after SchedulerOutput construction
- New concepts: `PauseState`, `use_v2_model_runner`, KV connector integration, encoder cache, spec decode scheduling
- Eviction hooks (`_process_evictions`, `update_request_mask`) do not exist in upstream v0.19
- `_free_request` method exists at line ~1560 for request cleanup
- `update_from_output` at line ~1090 handles post-step updates

### AUDIT-05: EngineCoreClient Subclasses (v0.19)
| Class | Line | Purpose |
|-------|------|---------|
| `EngineCoreClient` (ABC) | 69 | Abstract base with all method stubs |
| `InprocClient` | 274 | In-process for LLMEngine (no IPC) |
| `MPClient` | 460 | Base multi-process client (ZMQ) |
| `SyncMPClient` | 703 | Synchronous ZMQ client |
| `AsyncMPClient` | 874 | Async ZMQ client (main production path) |
| `DPAsyncMPClient` | 1124 | Data-parallel async client |
| `DPLBAsyncMPClient` | 1304 | Data-parallel with load balancing (NEW in v0.19) |

All subclasses need `update_request_mask_async` and `get_request_l2_norms_async` implementations. `DPLBAsyncMPClient` is new and not present in v0.14.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Python 3.12 | Runtime | Yes | 3.12.8 | -- |
| PyTorch | vLLM engine | Yes (venv) | 2.9.1+cu128 | -- |
| Pydantic | Protocol models | Yes (venv) | 2.12.5 | -- |
| pytest | Test runner | Yes (venv) | 9.0.2 | -- |
| CUDA/GPU | Smoke test, L2 norms | No (login node) | -- | Run on GPU compute nodes |
| nvidia-smi | GPU detection | No | -- | GPU compute nodes |
| git (upstream remote) | Audit v0.19 source | Yes | configured | -- |
| v0.19.0 tag | Audit source access | Yes | fetched as ref | -- |
| deepseek-8b model | Smoke test | Yes | on disk | -- |
| httpx/requests | Smoke test HTTP client | Check venv | -- | Use stdlib urllib |

**Missing dependencies with no fallback:**
- GPU/CUDA: Required for smoke test execution. Cannot run on login node. Plan must account for this.

**Missing dependencies with fallback:**
- None critical.

## Open Questions

1. **InprocClient -> EngineCore -> Scheduler access chain**
   - What we know: `InprocClient.engine_core` is an `EngineCore` instance. `EngineCore` has `self.scheduler`.
   - What's unclear: Whether `EngineCore` has a wrapper method for `update_request_mask` or if we go directly to the scheduler.
   - Recommendation: Check `EngineCore` class in v0.14 for existing `update_request_mask` method before implementing.

2. **Smoke test execution environment**
   - What we know: No GPU on login node. Model exists on disk at `$HOME/scratch/models/deepseek-8b`.
   - What's unclear: How to reliably run the smoke test (SLURM job? Interactive GPU session? CI pipeline?).
   - Recommendation: Write the test as a standard pytest file. Document execution instructions. Mark with appropriate pytest marker.

3. **DeepSeek model think block behavior**
   - What we know: DeepSeek R1-derived models produce `<think>...</think>` blocks during reasoning.
   - What's unclear: Whether `deepseek-8b` specifically (which appears to be DeepSeek-R1-Lite-8B or similar) reliably produces extended think blocks.
   - Recommendation: Use a math/reasoning prompt that strongly triggers thinking. Test manually first if possible.

## Sources

### Primary (HIGH confidence)
- `git show v0.19.0:vllm/v1/engine/__init__.py` -- EngineCoreRequestType byte values (AUDIT-01)
- `git show v0.19.0:vllm/sampling_params.py` -- SamplingParams field enumeration (AUDIT-02)
- `git show v0.19.0:vllm/v1/worker/block_table.py` -- BlockTable/MultiGroupBlockTable API (AUDIT-03)
- `git show v0.19.0:vllm/v1/core/sched/scheduler.py` -- schedule() structure (AUDIT-04)
- `git show v0.19.0:vllm/v1/engine/core_client.py` -- EngineCoreClient subclasses (AUDIT-05)
- `git show v0.19.0:vllm/v1/core/sched/output.py` -- SchedulerOutput fields

### Secondary (HIGH confidence)
- Direct source inspection of fork's current codebase (all bug fix targets verified at exact line numbers)
- `.planning/codebase/CONCERNS.md` -- pre-existing bug documentation
- `.planning/codebase/TESTING.md` -- test infrastructure patterns

## Metadata

**Confidence breakdown:**
- Bug fixes (CLEAN-01, CLEAN-02): HIGH -- exact lines identified, fixes are mechanical
- Cleanup items (D-11 to D-15): HIGH -- exact lines identified, changes are well-scoped
- Smoke test (CLEAN-03): MEDIUM -- design is clear but GPU availability adds execution uncertainty
- Audit findings (AUDIT-01 to AUDIT-05): HIGH -- all verified against real v0.19.0 tag (commit 2a69949)

**Research date:** 2026-04-08
**Valid until:** 2026-05-08 (v0.19.0 is a release tag, findings are stable)
