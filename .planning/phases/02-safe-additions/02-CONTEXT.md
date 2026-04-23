# Phase 2: Safe Additions - Context

**Gathered:** 2026-04-08
**Status:** Ready for planning

<domain>
## Phase Boundary

All eviction modules, custom fields, and IPC interface extensions exist in the v0.19 codebase with zero conflicts — the eviction code is present but not yet wired into runtime logic. This phase copies files, appends struct fields, and implements IPC methods. Nothing calls the new code yet (Phase 3-4 wires callers).

</domain>

<decisions>
## Implementation Decisions

### IPC Method Depth
- **D-01:** Full implementation for all 5 client subclasses (`InprocClient`, `SyncMPClient`, `AsyncMPClient`, `DPAsyncMPClient`, `DPLBAsyncMPClient`). Methods are callable end-to-end with complete serialize/send/dispatch logic. Nothing invokes them until Phase 3-4 wires the callers.
- **D-02:** This includes `WorkerBase` ABC declarations (IPC-01), `GPUWorker` implementations (IPC-02), all client methods (IPC-03), `AsyncLLM` methods (IPC-04), and `EngineCore`/`EngineCoreProc` dispatch (IPC-05).

### Module Adaptation
- **D-03:** Adapt imports and references during copy — every copied file must be import-clean against v0.19 on arrival. No known-broken intermediate state.
- **D-04:** Use the Phase 1-cleaned versions of files (e.g., `l2_norm_cache.py` with dual singleton fixed, `flash_attn.py` with dead fields removed).

### Plan Granularity
- **D-05:** Two plans for Phase 2. Plan 1: MERGE-01..08 (copy modules, add fields/types). Plan 2: IPC-01..05 (worker ABC, client methods, engine dispatch). Natural dependency boundary — IPC needs the types from MERGE.

### Test Handling
- **D-06:** Copy all existing test files from `tests/eviction/` to v0.19 base with imports adapted to match v0.19 paths. Source-inspect tests may still fail until Phase 3-4 modifies target files, but infrastructure is present and import-clean.
- **D-07:** Include the functional smoke test (`test_smoke.py`) in the copy.

### Field Positioning (from Phase 1 audit)
- **D-08:** `enable_l2_norms` and `l2_norm_layers` appended AFTER `repetition_detection` (last v0.19 upstream field) and BEFORE internal post_init fields (`output_text_buffer_length`, `_eos_token_id`, `_all_stop_token_ids`) in `SamplingParams`.
- **D-09:** `new_l2_norms` appended as last field on `EngineCoreOutput`.
- **D-10:** `evictable_token_ranges_map` added to `SchedulerOutput` dataclass.
- **D-11:** `new_l2_norms` added to `RequestOutput`.
- **D-12:** `UPDATE_MASK` assigned to `b"\x06"` (first non-colliding byte after v0.19's `WAKEUP = b"\x05"`).

### Claude's Discretion
- Exact import path fixups needed during module adaptation (determined by reading v0.19 source)
- Internal ordering of work within each plan
- Commit granularity within plans

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase 1 Audit Findings (critical for field positioning and byte values)
- `.planning/phases/01-groundwork-audit/01-AUDIT-FINDINGS.md` — All 5 audit results: byte collision confirmed, SamplingParams field layout, InputBatch block table API, schedule() structure, client subclass hierarchy

### Eviction Module Source (v0.14 — files to copy)
- `vllm/thought_eviction/orchestrator.py` — EvictionOrchestrator, stream wrapping
- `vllm/thought_eviction/segmenter.py` — ThoughtSegmenter, ThoughtSegment
- `vllm/thought_eviction/strategies.py` — All four strategy classes
- `vllm/thought_eviction/block_utils.py` — Range merge, block-align utilities
- `vllm/thought_eviction/__init__.py` — Package init
- `vllm/v1/attention/l2_norm_cache.py` — L2NormCache (Phase 1-cleaned version)

### v0.19 Target Files (fields/types to modify)
- `vllm/sampling_params.py` — Append `enable_l2_norms`, `l2_norm_layers` (MERGE-04)
- `vllm/v1/engine/__init__.py` — Append `new_l2_norms` to `EngineCoreOutput`, add `UPDATE_MASK` to `EngineCoreRequestType` (MERGE-05, MERGE-06)
- `vllm/v1/core/sched/output.py` — Add `evictable_token_ranges_map` to `SchedulerOutput` (MERGE-07)
- `vllm/outputs.py` — Add `new_l2_norms` to `RequestOutput` (MERGE-08)

### v0.19 IPC Layer (methods to add)
- `vllm/v1/engine/core_client.py` — All 5 client subclasses (IPC-03), ABC at line 69
- `vllm/v1/worker/gpu_worker.py` — `GPUWorker` (IPC-02)
- `vllm/v1/engine/async_llm.py` — `AsyncLLM` (IPC-04)
- `vllm/v1/engine/core.py` — `EngineCore`/`EngineCoreProc` dispatch (IPC-05)

### Existing Tests (to copy)
- `tests/eviction/` — All existing eviction test files
- `tests/eviction/test_smoke.py` — Functional smoke test (Phase 1)

### Codebase Analysis
- `.planning/codebase/ARCHITECTURE.md` — Layer architecture and data flow
- `.planning/codebase/STRUCTURE.md` — File layout and modification map

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `eviction_changes.patch` — Complete diff of all fork changes vs upstream; useful for identifying exact code to port
- Phase 1-cleaned `l2_norm_cache.py` — Dual singleton fixed, dead `update_norms()` removed
- Phase 1-cleaned `flash_attn.py` — Dead `compute_l2_norms`/`request_ids` fields removed

### Established Patterns
- `import regex as re` required (not `import re`) — enforced by pre-commit
- msgspec structs use `omit_defaults=True` with `array_like` semantics — field ordering is byte-significant for IPC
- `EngineCoreRequestType` uses single-byte enum values over ZMQ protocol
- Client hierarchy: `EngineCoreClient` ABC → `InprocClient` (direct) / `MPClient` → `SyncMPClient` / `AsyncMPClient` → `DPAsyncMPClient` → `DPLBAsyncMPClient`

### Integration Points
- Eviction fields on `SamplingParams` must go after `repetition_detection` (line 291 in v0.19)
- `SchedulerOutput` constructed at line 914 in v0.19 `scheduler.py`
- `EngineCoreRequestType` enum at lines 217-230 in v0.19 `__init__.py`
- `WorkerBase` ABC needs abstract method declarations for eviction

</code_context>

<specifics>
## Specific Ideas

- `UPDATE_MASK` byte value: exactly `b"\x06"` (next free after v0.19's `WAKEUP = b"\x05"`)
- Use Phase 1-cleaned versions of modified files (not original v0.14 versions with known bugs)

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 02-safe-additions*
*Context gathered: 2026-04-08*
