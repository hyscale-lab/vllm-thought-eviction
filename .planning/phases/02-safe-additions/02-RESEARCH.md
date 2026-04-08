# Phase 2: Safe Additions - Research

**Researched:** 2026-04-08
**Domain:** vLLM v0.14->v0.19 merge reconciliation + eviction code integration
**Confidence:** HIGH

## Summary

Phase 2's scope is fundamentally different from what the description implies. The current `upgrade_vllm` branch already contains all eviction modules, fields, and IPC methods from the v0.14 fork -- they were carried forward during the merge. However, the merge resolution incorrectly picked v0.14 versions of many shared files, **removing v0.19 features** and **using wrong byte values**. Phase 2 is therefore a **reconciliation phase**: restore accidentally-removed v0.19 features, fix field positioning to comply with D-08/D-09/D-10, fix the UPDATE_MASK byte collision, and ensure eviction code is import-clean against v0.19.

The critical risk is that the branch currently has a broken state where v0.19 features like `WAKEUP` shutdown signaling, `RepetitionDetectionParams`, `PauseMode`, `EEPNotificationType`, and `new_block_ids_to_zero` were removed during merge resolution. These must be restored alongside the eviction additions.

**Primary recommendation:** Treat this phase as "reconcile merge conflicts correctly" rather than "add new files." Start from v0.19.0 as the canonical source for each target file, then surgically add eviction code on top.

<user_constraints>

## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01:** Full implementation for all 5 client subclasses (`InprocClient`, `SyncMPClient`, `AsyncMPClient`, `DPAsyncMPClient`, `DPLBAsyncMPClient`). Methods are callable end-to-end with complete serialize/send/dispatch logic. Nothing invokes them until Phase 3-4 wires the callers.
- **D-02:** This includes `WorkerBase` ABC declarations (IPC-01), `GPUWorker` implementations (IPC-02), all client methods (IPC-03), `AsyncLLM` methods (IPC-04), and `EngineCore`/`EngineCoreProc` dispatch (IPC-05).
- **D-03:** Adapt imports and references during copy -- every copied file must be import-clean against v0.19 on arrival. No known-broken intermediate state.
- **D-04:** Use the Phase 1-cleaned versions of files (e.g., `l2_norm_cache.py` with dual singleton fixed, `flash_attn.py` with dead fields removed).
- **D-05:** Two plans for Phase 2. Plan 1: MERGE-01..08 (copy modules, add fields/types). Plan 2: IPC-01..05 (worker ABC, client methods, engine dispatch).
- **D-06:** Copy all existing test files from `tests/eviction/` to v0.19 base with imports adapted to match v0.19 paths.
- **D-07:** Include the functional smoke test (`test_smoke.py`) in the copy.
- **D-08:** `enable_l2_norms` and `l2_norm_layers` appended AFTER `repetition_detection` (last v0.19 upstream field) and BEFORE internal post_init fields (`output_text_buffer_length`, `_eos_token_id`, `_all_stop_token_ids`) in `SamplingParams`.
- **D-09:** `new_l2_norms` appended as last field on `EngineCoreOutput`.
- **D-10:** `evictable_token_ranges_map` added to `SchedulerOutput` dataclass.
- **D-11:** `new_l2_norms` added to `RequestOutput`.
- **D-12:** `UPDATE_MASK` assigned to `b"\x06"` (first non-colliding byte after v0.19's `WAKEUP = b"\x05"`).

### Claude's Discretion
- Exact import path fixups needed during module adaptation (determined by reading v0.19 source)
- Internal ordering of work within each plan
- Commit granularity within plans

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope

</user_constraints>

<phase_requirements>

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| MERGE-01 | Copy `vllm/thought_eviction/` module to v0.19 base | Files already exist on branch but need import fixes (`import re` -> `import regex as re`) |
| MERGE-02 | Copy `vllm/v1/attention/l2_norm_cache.py` to v0.19 base | File already exists; Phase 1-cleaned version in place |
| MERGE-03 | Copy `tests/eviction/` to v0.19 base | Files already exist; need import adaptation check |
| MERGE-04 | Append `enable_l2_norms` and `l2_norm_layers` to `SamplingParams` | Fields exist but at WRONG position (after `skip_clone`); must be after `repetition_detection`. Also v0.19 fields `thinking_token_budget`, `repetition_detection`, `_eos_token_id` are missing from branch |
| MERGE-05 | Append `new_l2_norms` to `EngineCoreOutput` | Field exists and correctly positioned as last field. But v0.19 features `PauseMode`, `EEPNotificationType`, `REPETITION`, `resumable`, `reasoning_ended`, `num_external_computed_tokens` are missing |
| MERGE-06 | Add `UPDATE_MASK` to `EngineCoreRequestType` | EXISTS but at `b'\x05'` (collision with WAKEUP). Must change to `b'\x06'` and restore `WAKEUP = b'\x05'` |
| MERGE-07 | Add `evictable_token_ranges_map` to `SchedulerOutput` | Field exists but v0.19's `new_block_ids_to_zero` was REMOVED; must restore |
| MERGE-08 | Add `new_l2_norms` to `RequestOutput` | Field exists and correctly positioned |
| IPC-01 | Add eviction abstract methods to `WorkerBase` ABC | Methods exist: `get_request_l2_norms`, `configure_l2_norms` |
| IPC-02 | Implement eviction methods on `GPUWorker` | Methods exist: `get_request_l2_norms`, `configure_l2_norms` |
| IPC-03 | Add eviction methods to all `EngineCoreClient` subclasses | Methods exist on ABC, `InprocClient`, `SyncMPClient`, `AsyncMPClient`. `DPAsyncMPClient`/`DPLBAsyncMPClient` inherit from `AsyncMPClient` |
| IPC-04 | Add `update_request_mask` and `get_request_l2_norms` to `AsyncLLM` | Methods exist |
| IPC-05 | Add `UPDATE_MASK` dispatch to `EngineCore`/`EngineCoreProc` | Dispatch exists in `_handle_client_request` but WAKEUP handler was removed |

</phase_requirements>

## Critical Finding: Branch State Regression

The `upgrade_vllm` branch was created from the v0.14 fork (which had eviction code), then v0.19 was merged in. Merge resolution incorrectly chose v0.14 versions of shared files in many cases, **removing v0.19 features**. This is the primary technical challenge of Phase 2.

### Removed v0.19 Features (Must Restore)

**Confidence: HIGH** -- verified via `git diff v0.19.0`

#### `vllm/v1/engine/__init__.py`
| Feature | v0.19 Has | Branch Has | Impact |
|---------|-----------|------------|--------|
| `PauseMode` type alias | Yes | **NO** | Breaks pause_generation API |
| `EEP_NOTIFICATION_CALL_ID` | Yes | **NO** | Breaks elastic engine protocol |
| `EEPNotificationType` enum | Yes | **NO** | Breaks elastic engine protocol |
| `FINISH_REASON_STRINGS` with "repetition" | Yes | **NO** | Breaks repetition detection |
| `FinishReason.REPETITION = 4` | Yes | **NO** | Breaks repetition detection |
| `WAKEUP = b"\x05"` | Yes | **NO** (replaced by UPDATE_MASK) | **Breaks shutdown signaling** |
| `resumable` field on `EngineCoreRequest` | Yes | **NO** | Breaks pause/resume |
| `reasoning_ended` field on `EngineCoreRequest` | Yes | **NO** | Breaks thinking budget |
| `num_external_computed_tokens` on `EngineCoreOutput` | Yes | **NO** | Breaks KV connector stats |
| `new_data_parallel_master_port_list` on `ReconfigureDistributedRequest` | Yes | **NO** | Breaks DP reconfigure |
| `coord_store_port` on `ReconfigureDistributedRequest` | Yes | **NO** | Breaks DP reconfigure |
| `eos_token_id` removed from `EngineCoreRequest` (v0.19 moved it) | Removed | **Present** | Duplicate/stale field |

#### `vllm/sampling_params.py`
| Feature | v0.19 Has | Branch Has | Impact |
|---------|-----------|------------|--------|
| `RepetitionDetectionParams` class | Yes | **NO** | Breaks repetition detection |
| `thinking_token_budget` field | Yes | **NO** | Breaks thinking budget feature |
| `repetition_detection` field | Yes | **NO** | Breaks repetition detection |
| `_eos_token_id` field | Yes | **NO** | Breaks EOS handling |
| `logits_processors` field | **NO** (removed) | Yes | Stale v0.14 field |
| `truncate_prompt_tokens` field | **NO** (removed) | Yes | Stale v0.14 field |
| `disable_fallback` on `StructuredOutputsParams` | **NO** | Yes | Stale v0.14 field |
| `n` field docs with VLLM_MAX_N_SEQUENCES | Yes | **NO** | Missing docs |

#### `vllm/v1/core/sched/output.py`
| Feature | v0.19 Has | Branch Has | Impact |
|---------|-----------|------------|--------|
| `new_block_ids_to_zero` field on `SchedulerOutput` | Yes | **NO** | **Breaks KV cache zeroing** |
| `@bc_linter_include` decorators | **NO** | Yes | Harmless but noisy |

#### `vllm/v1/engine/core.py`
| Feature | v0.19 Has | Branch Has | Impact |
|---------|-----------|------------|--------|
| `WAKEUP` handler in `_handle_client_request` | Yes | **NO** | **Breaks shutdown** |
| Signal callback with `wakeup_engine` | Yes | **NO** | **Breaks signal handling** |

### Import Violations in Eviction Code

**Confidence: HIGH** -- verified via grep

| File | Issue | Fix |
|------|-------|-----|
| `vllm/thought_eviction/orchestrator.py` | `import re` | Must be `import regex as re` |
| `vllm/thought_eviction/segmenter.py` | `import re` | Must be `import regex as re` |

These will fail the `enforce-import-regex-instead-of-re` pre-commit hook.

### UPDATE_MASK Byte Collision

**Confidence: HIGH** -- verified via `git show v0.19.0` and branch state

Current branch: `UPDATE_MASK = b'\x05'` (no WAKEUP)
v0.19 upstream: `WAKEUP = b'\x05'` (no UPDATE_MASK)
Required: `WAKEUP = b'\x05'` AND `UPDATE_MASK = b'\x06'`

The WAKEUP sentinel is used internally by `EngineCoreProc` to wake up the input queue during shutdown. Without it, the engine core process cannot be cleanly shut down via signal handlers.

## Architecture Patterns

### Pattern 1: Restoring v0.19 Base Then Adding Eviction

**What:** For each target file, start from the v0.19.0 version (`git show v0.19.0:<path>`), then surgically add eviction code on top.

**Why:** The current branch state is unreliable -- it's unclear which changes are intentional eviction additions vs accidental merge regressions. Using v0.19.0 as ground truth eliminates this ambiguity.

**Process:**
1. Read v0.19.0 version of target file
2. Read current branch version to identify eviction additions
3. Write corrected version = v0.19.0 base + eviction additions

### Pattern 2: msgspec Field Ordering (Critical for IPC)

**What:** `SamplingParams` and `EngineCoreOutput` use `msgspec.Struct` with `array_like=True, omit_defaults=True`. Field ordering is byte-significant for serialization.

**Rule:** Eviction fields must always come AFTER all upstream fields. For `SamplingParams`, this means after `repetition_detection` and before `output_text_buffer_length`. For `EngineCoreOutput`, after `num_nans_in_logits`.

**Source:** Phase 1 audit (AUDIT-02), CONTEXT.md decisions D-08, D-09

### Pattern 3: Client Hierarchy Inheritance

**What:** `DPAsyncMPClient` and `DPLBAsyncMPClient` inherit from `AsyncMPClient`. If `AsyncMPClient` has correct eviction method implementations, the DP clients inherit them automatically.

**Caveat:** `DPLBAsyncMPClient` overrides `call_utility_async` to fan out to all engines. If `get_request_l2_norms` uses `call_utility_async`, it will automatically work correctly in DP mode (fan-out, return first result).

**Caveat:** `update_request_mask_async` on `AsyncMPClient` sends via `_send_input` with `UPDATE_MASK_REQUEST_TYPE`. In DP mode, `DPAsyncMPClient` may need to route to the correct engine. This requires checking whether `_send_input` is overridden or if requests are routed per-engine.

### Anti-Patterns to Avoid

- **Removing v0.19 features to "simplify"**: Every removed feature breaks some v0.19 functionality. All upstream features must be preserved.
- **Using `import re`**: Must use `import regex as re` per pre-commit hook enforcement.
- **Adding fields in wrong position**: msgspec array_like encoding makes field order critical. Wrong position = silent IPC data corruption.
- **Replacing WAKEUP with UPDATE_MASK**: Both must coexist. UPDATE_MASK gets the next free byte (`b"\x06"`).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| v0.19 file restoration | Manual line-by-line reconstruction | `git show v0.19.0:<path>` as source of truth | Ensures no v0.19 features are missed |
| Byte value assignment | Guessing free values | Phase 1 AUDIT-01 findings | Collision at b'\x05' already verified |
| Field positioning | Inserting by line number | Structural placement relative to named fields | Line numbers shift; field names are stable |

## Common Pitfalls

### Pitfall 1: Silently Removing v0.19 Features
**What goes wrong:** Merge resolution picks v0.14 version, removing v0.19-only features (PauseMode, WAKEUP, RepetitionDetectionParams, etc.)
**Why it happens:** v0.14 fork modified the same files that v0.19 changed; merge conflict resolution is manual and error-prone
**How to avoid:** Always diff against v0.19.0 tag after modifying a file; verify no v0.19 features are missing
**Warning signs:** `git diff v0.19.0 -- <file>` shows lines being removed that aren't eviction-related

### Pitfall 2: UPDATE_MASK Byte Collision
**What goes wrong:** UPDATE_MASK at b'\x05' collides with WAKEUP; engine misinterprets eviction commands as shutdown signals
**Why it happens:** v0.14 fork assigned b'\x05' before v0.19 added WAKEUP at the same value
**How to avoid:** Assign UPDATE_MASK to b'\x06' (D-12); keep WAKEUP at b'\x05'
**Warning signs:** Engine shutdown during eviction, or eviction silently failing

### Pitfall 3: SamplingParams Field Order Corruption
**What goes wrong:** IPC serialization/deserialization produces garbled values because fields are in wrong order
**Why it happens:** `array_like=True` encodes by position, not by name; adding fields in the middle shifts all subsequent positions
**How to avoid:** Per D-08, eviction fields go after `repetition_detection`, before `output_text_buffer_length`
**Warning signs:** Sampling parameters have unexpected values on the worker side; model produces garbage output

### Pitfall 4: Missing SchedulerOutput Fields
**What goes wrong:** `new_block_ids_to_zero` removal breaks KV cache memory zeroing; stale data corrupts attention
**Why it happens:** Branch replaced `new_block_ids_to_zero` with `evictable_token_ranges_map` instead of adding alongside
**How to avoid:** Both fields must exist on `SchedulerOutput`
**Warning signs:** NaN in attention outputs, memory corruption, CUDA errors

### Pitfall 5: WAKEUP Removal Breaks Shutdown
**What goes wrong:** Engine core process cannot be cleanly shut down; hangs on `input_queue.get()`
**Why it happens:** WAKEUP sentinel was removed from enum and its handler in `_handle_client_request` was removed
**How to avoid:** Restore WAKEUP at b'\x05' in enum and its no-op handler in `_handle_client_request`
**Warning signs:** Server hangs on Ctrl+C, engine process becomes zombie

## Code Examples

### Correct EngineCoreRequestType (after fix)
```python
class EngineCoreRequestType(enum.Enum):
    ADD = b"\x00"
    ABORT = b"\x01"
    START_DP_WAVE = b"\x02"
    UTILITY = b"\x03"
    # Sentinel used within EngineCoreProc.
    EXECUTOR_FAILED = b"\x04"
    # Sentinel to wake up input_queue.get() during shutdown.
    WAKEUP = b"\x05"
    # Eviction mask update (thought eviction).
    UPDATE_MASK = b"\x06"
```

### Correct SamplingParams Field Order (after fix)
```python
# ... (all v0.19 upstream fields) ...
skip_reading_prefix_cache: bool | None = None
thinking_token_budget: int | None = None
repetition_detection: RepetitionDetectionParams | None = None
# --- Eviction fields (D-08) ---
enable_l2_norms: bool = False
l2_norm_layers: list[int] | None = None
# --- Internal post_init fields ---
output_text_buffer_length: int = 0
_eos_token_id: int | None = None
_all_stop_token_ids: set[int] = msgspec.field(default_factory=set)
```

### Correct SchedulerOutput (both fields present)
```python
@dataclass
class SchedulerOutput:
    # ... (all existing fields) ...
    ec_connector_metadata: ECConnectorMetadata | None = None
    new_block_ids_to_zero: list[int] | None = None  # v0.19 upstream
    evictable_token_ranges_map: dict[str, list[tuple[int, int]]] | None = None  # eviction
```

### Correct _handle_client_request dispatch (both handlers)
```python
def _handle_client_request(self, request_type, request):
    if request_type == EngineCoreRequestType.WAKEUP:
        return  # v0.19 shutdown signal - no-op
    elif request_type == EngineCoreRequestType.ADD:
        req, request_wave = request
        self.add_request(req, request_wave)
    elif request_type == EngineCoreRequestType.ABORT:
        self.abort_requests(request)
    elif request_type == EngineCoreRequestType.UPDATE_MASK:
        request_id, ranges = request
        self.update_request_mask(request_id, ranges)
    elif request_type == EngineCoreRequestType.UTILITY:
        # ... utility handling ...
```

## State of the Art

| Old Approach (v0.14 fork) | Current Approach (v0.19) | Impact on Eviction |
|---------------------------|--------------------------|-------------------|
| `UPDATE_MASK = b'\x05'` | `WAKEUP = b'\x05'` at that byte | Must use `b'\x06'` |
| Direct `block_table` access | `MultiGroupBlockTable` wrapper | Phase 3 concern (GPU-01) |
| `logits_processors` field on SamplingParams | Removed; built from structured params | Remove from branch |
| `truncate_prompt_tokens` on SamplingParams | Removed | Remove from branch |
| No `RepetitionDetectionParams` | New class + field | Restore for v0.19 compat |
| No `thinking_token_budget` | New field | Restore for v0.19 compat |
| No `PauseMode` | New type alias | Restore for v0.19 compat |

## File-by-File Reconciliation Map

This is the core deliverable for the planner -- each file's required changes.

### Plan 1 Files (MERGE-01..08)

| File | Current State | Action Required |
|------|---------------|-----------------|
| `vllm/thought_eviction/orchestrator.py` | Exists, `import re` | Fix to `import regex as re` |
| `vllm/thought_eviction/segmenter.py` | Exists, `import re` | Fix to `import regex as re` |
| `vllm/thought_eviction/strategies.py` | Exists, clean | Verify imports only |
| `vllm/thought_eviction/block_utils.py` | Exists, clean | Verify imports only |
| `vllm/thought_eviction/__init__.py` | Exists | Verify imports only |
| `vllm/v1/attention/l2_norm_cache.py` | Exists, Phase 1-cleaned | Verify clean |
| `tests/eviction/*` | Exists (10 test files + smoke) | Verify imports |
| `vllm/sampling_params.py` | Missing v0.19 fields, wrong eviction field position | Restore v0.19 base, reposition eviction fields per D-08 |
| `vllm/v1/engine/__init__.py` | Missing v0.19 features, wrong UPDATE_MASK byte | Restore v0.19 base, add eviction fields per D-09/D-12 |
| `vllm/v1/core/sched/output.py` | Missing `new_block_ids_to_zero` | Restore v0.19 base, add eviction field per D-10 |
| `vllm/outputs.py` | `new_l2_norms` correctly added | Verify; check for other regressions |

### Plan 2 Files (IPC-01..05)

| File | Current State | Action Required |
|------|---------------|-----------------|
| `vllm/v1/worker/worker_base.py` | Eviction ABCs exist | Verify correct |
| `vllm/v1/worker/gpu_worker.py` | Eviction methods exist | Verify correct |
| `vllm/v1/engine/core_client.py` | Eviction methods on ABC + 3 clients | Verify DP clients inherit correctly; check for v0.19 regressions |
| `vllm/v1/engine/async_llm.py` | Eviction methods exist | Verify correct |
| `vllm/v1/engine/core.py` | UPDATE_MASK dispatch exists, WAKEUP removed | Restore WAKEUP handler; verify no other regressions |

## Open Questions

1. **How deep are the v0.19 regressions in core.py and core_client.py?**
   - What we know: `__init__.py`, `sampling_params.py`, `output.py` have confirmed regressions
   - What's unclear: The 1231-line diff in `core.py` and 908-line diff in `core_client.py` may contain many more regressions
   - Recommendation: Full diff audit of these files before planning task details

2. **Are there regressions in outputs.py beyond the eviction additions?**
   - What we know: `new_l2_norms` added correctly, `multi_modal_placeholders` added (not in v0.19)
   - What's unclear: Whether `multi_modal_placeholders` is a v0.14 artifact or intentional
   - Recommendation: Check via `git show v0.19.0:vllm/outputs.py`

3. **Do DPAsyncMPClient/DPLBAsyncMPClient need explicit overrides?**
   - What we know: They inherit from AsyncMPClient which has eviction methods
   - What's unclear: Whether `DPLBAsyncMPClient.call_utility_async` fan-out is correct for `get_request_l2_norms`
   - Recommendation: Inherited methods are likely sufficient; verify in implementation

## Sources

### Primary (HIGH confidence)
- `git show v0.19.0:<path>` commands -- verified exact v0.19.0 state of all target files
- `git diff v0.19.0 -- <path>` commands -- verified all regressions against v0.19.0 tag
- `.planning/phases/01-groundwork-audit/01-AUDIT-FINDINGS.md` -- Phase 1 audit verified findings

### Secondary (MEDIUM confidence)
- Current branch file inspection -- may contain undocumented intentional changes
- CONTEXT.md decisions -- user intent but implementation details are discretionary

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH - no new libraries; pure code reconciliation
- Architecture: HIGH - all patterns established in Phase 1 audit and v0.19.0 source
- Pitfalls: HIGH - verified via direct git diff against v0.19.0 tag; every regression confirmed

**Research date:** 2026-04-08
**Valid until:** No expiry (pinned to v0.19.0 commit 2a69949)
