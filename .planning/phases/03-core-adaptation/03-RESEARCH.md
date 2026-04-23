# Phase 3: Core Adaptation - Research

**Researched:** 2026-04-08
**Domain:** vLLM v0.19 scheduler and GPU model runner eviction integration
**Confidence:** HIGH

## Summary

Phase 3 re-applies thought eviction logic to two files that are currently pure upstream v0.19 code: the scheduler (`vllm/v1/core/sched/scheduler.py`, 2317 lines) and the GPU model runner (`vllm/v1/worker/gpu_model_runner.py`, 7059 lines). The pre-merge reference (commit `2ec9a65e84b`) contains proven working eviction code that must be ported with API adaptations to v0.19 structures.

The v0.19 APIs are well-understood from Phase 1 audit findings and direct code inspection. Key adaptation points are: (1) `update_from_output` now returns `dict[int, EngineCoreOutputs]` keyed by `client_index` instead of a flat structure, (2) block tables are accessed via `MultiGroupBlockTable[0]` instead of a flat block table, (3) KV cache freeing goes through `kv_cache_manager.free_blocks()` which delegates to the coordinator hierarchy, and (4) `SchedulerOutput` construction already has the `evictable_token_ranges_map` field (auto-merged in Phase 2.1). All supporting infrastructure (L2NormCache, EngineCoreOutput.new_l2_norms, SamplingParams.enable_l2_norms, IPC layer) is already in place from Phases 2 and 2.1.

**Primary recommendation:** Port eviction code method-by-method from pre-merge reference, adapting each to v0.19 APIs. Scheduler first (simpler, fewer API changes), then GPU runner (more complex block table and attn_metadata adaptations).

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- D-01: Port from pre-merge reference (commit `2ec9a65e84b`), adapting each piece to v0.19 APIs
- D-02: Both scheduler AND GPU model runner adaptation in Phase 3
- D-03: Verification level: import + unit tests
- D-04: Native adaptation -- rewrite eviction code to use v0.19 APIs directly. No shims or wrappers
- D-05: L2 norm retrieval inside existing `update_from_output` output loop, adapted to `dict[int, EngineCoreOutputs]` return structure
- D-06: Block table invalidation targets KV cache group 0 only via `MultiGroupBlockTable[0]`
- D-07: KV cache block freeing calls `self.kv_cache_manager.free_blocks(request_id, block_indices)`
- D-08: Add `request_eviction_data` and `_l2_norm_last_index` dicts to `Scheduler.__init__`
- D-09: Add `update_request_mask()` method on scheduler
- D-10: Add `_process_evictions()` method with `kv_cache_manager.free_blocks()` calls
- D-11: Add `_process_evictions()` call in `schedule()` method
- D-12: Add L2 norm retrieval in `update_from_output`
- D-13: Add eviction cleanup in `_free_request()`
- D-14: Add `_compute_l2_norms()` method in GPU runner
- D-15: Add L2 norm cache initialization in GPU runner `__init__`
- D-16: Add block table invalidation using `self.input_batch.block_table[0]`
- D-17: Add `_compute_l2_norms()` call in `execute_model`
- D-18: Add request cleanup: `self.l2_norm_cache.remove_request(req_id)` on request finish
- D-19: Update source-inspect tests to match new v0.19-adapted function signatures
- D-20: Run existing `tests/eviction/` suite only -- no new mock integration tests
- D-21: Functional smoke test stays in Phase 4 scope

### Claude's Discretion
- Exact insertion points for eviction code in scheduler.py and gpu_model_runner.py
- Commit granularity within Phase 3 plans
- Plan count and ordering (scheduler-first vs GPU-first vs parallel)
- Whether to add lightweight integration checks beyond import + unit tests
- Specific adaptations needed for each pre-merge function when porting to v0.19

### Deferred Ideas (OUT OF SCOPE)
None -- discussion stayed within phase scope
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| SCHED-01 | Add `request_eviction_data` and `_l2_norm_last_index` dicts to v0.19 Scheduler | Insertion point: after line 182 in `__init__`, after `failed_recving_kv_req_ids` |
| SCHED-02 | Re-apply `update_request_mask` method on scheduler | Port from pre-merge lines 258-267; no API changes needed |
| SCHED-03 | Re-apply `_process_evictions` method with correct placement in v0.19 `schedule()` | Port from pre-merge lines 269-313; `kv_cache_manager.free_blocks()` API confirmed compatible; insert `_process_evictions()` call at top of `schedule()` (line 360), pass result to `SchedulerOutput` at line 930 |
| SCHED-04 | Adapt L2 norm retrieval in `update_from_output` for multi-client dict return type | L2 norms go inside per-request loop (around line 1456), before `EngineCoreOutput` construction at line 1464; `new_l2_norms=new_l2_norms` kwarg added to constructor |
| SCHED-05 | Add `free_blocks` support through v0.19 KV cache coordinator hierarchy | `kv_cache_manager.free_blocks(request_id, block_indices)` confirmed at line 430 of kv_cache_manager.py, delegates to coordinator.free_blocks at line 191 |
| SCHED-06 | Verify all `SchedulerOutput` construction sites pass `evictable_token_ranges_map` | Two sites: main construction (line 914) and `make_empty` (line 247). Main site needs `evictable_token_ranges_map=processed_ranges or None`; `make_empty` already omits it (defaults to None) |
| GPU-01 | Adapt block table invalidation to v0.19 `InputBatch.block_table` API | `self.input_batch.block_table[0]` returns `BlockTable` for group 0; access `.block_table.np` for numpy array, `.block_size` for block size |
| GPU-02 | Re-apply `_compute_l2_norms` method in v0.19 model runner | Port from pre-merge lines 1224-1268; `attn_metadata` type changed to `PerLayerAttnMetadata` (dict or list of dicts); `self.kv_caches` still available at line 494 |
| GPU-03 | Re-apply L2 norm cache initialization and KV replacement strategy dispatch | Add after line 854 in `__init__`: `self.l2_norm_cache = get_l2_norm_cache()`, strategy map, `self.evicted_ranges` dict |
| GPU-04 | Re-apply `execute_model` L2 norm computation call | Insert after `model_output = self._model_forward(...)` at line 4029, before postprocess block at line 4031 |
| GPU-05 | Remove dead `compute_l2_norms`/`request_ids` fields from FlashAttentionMetadata | Already clean -- no eviction fields exist in current v0.19 FlashAttentionMetadata. This requirement is already satisfied by the Phase 2.1 upstream wholesale accept |
</phase_requirements>

## Architecture Patterns

### Scheduler Insertion Map

The scheduler has 6 insertion points, all precisely located:

```
scheduler.py (2317 lines, pure upstream)

Line 68:   class Scheduler
Line 78:     def __init__
Line 182:      self.failed_recving_kv_req_ids  <-- INSERT eviction dicts after here
Line 348:    def schedule()
Line 360:      scheduled_new_reqs = ...        <-- INSERT _process_evictions() call before here
Line 914:      scheduler_output = SchedulerOutput(
Line 930:        )                              <-- ADD evictable_token_ranges_map= kwarg
Line 1302:   def update_from_output()
Line 1456:     if new_token_ids or ...          <-- INSERT L2 norm retrieval before EngineCoreOutput
Line 1464:       EngineCoreOutput(               <-- ADD new_l2_norms=new_l2_norms kwarg
Line 1823:   def _free_request()
Line 1843:     del self.requests[...]           <-- INSERT eviction cleanup before here
```

### GPU Model Runner Insertion Map

The GPU runner has 5 insertion points:

```
gpu_model_runner.py (7059 lines, pure upstream)

Line 394:    def __init__
Line 854:      self.layerwise_nvtx_hooks_registered  <-- INSERT l2_norm_cache, strategy map, evicted_ranges after here
Line 1049:   def _update_states()
Line 1060:     for req_id in finished_req_ids:       <-- INSERT evicted_ranges cleanup, l2_norm_cache.remove_request
Line 1356:     self.input_batch.refresh_metadata()    <-- INSERT eviction block invalidation after here
Line 3759:   def execute_model()
Line 4029:     model_output = self._model_forward()   <-- INSERT _compute_l2_norms call after here
```

Additionally, 3 new methods to add to GPUModelRunner:
- `_compute_l2_norms(self, attn_metadata)` -- port from pre-merge
- `_replace_kv_caches_sink(...)` -- port from pre-merge
- `_replace_kv_caches_zero(...)` -- port from pre-merge
- `_replace_kv_caches_nearby(...)` -- port from pre-merge

### Key API Adaptations Required

| Pre-merge Pattern | v0.19 Adaptation | Risk |
|-------------------|-------------------|------|
| Flat block table access | `self.input_batch.block_table[0]` for group 0 | LOW -- `__getitem__` confirmed |
| `block_table.block_table.np` | Same pattern, accessed via `BlockTable.block_table.np` | LOW -- verified in block_table.py |
| `attn_metadata_dict: dict[str, AttentionMetadata]` | `PerLayerAttnMetadata` which is `dict[str, AttentionMetadata]` or `list[dict[str, AttentionMetadata]]` | MEDIUM -- ubatch case returns list |
| `update_from_output` building flat outputs | Now builds `dict[int, list[EngineCoreOutput]]` keyed by `client_index` | LOW -- L2 norms go inside per-request loop, unaffected by keying |
| `self.kv_cache_manager.free_blocks(req_id, blocks)` | Same API, line 430 in kv_cache_manager.py | LOW -- signature matches |
| `SchedulerOutput(..., evictable_token_ranges_map=...)` | Field already exists (line 243 in output.py) | LOW -- just add kwarg |
| `EngineCoreOutput(..., new_l2_norms=...)` | Field already exists (line 172 in __init__.py), added `num_external_computed_tokens` kwarg in v0.19 | LOW -- add after existing kwargs |

### attn_metadata Adaptation for _compute_l2_norms

The pre-merge `_compute_l2_norms` takes `attn_metadata_dict: dict[str, AttentionMetadata]`. In v0.19, `attn_metadata` in `execute_model` is typed as `PerLayerAttnMetadata` which is `dict[str, AttentionMetadata] | list[dict[str, AttentionMetadata]]`.

For the non-ubatch case (common case), it is a plain dict -- same as pre-merge. For the ubatch case, it is a list of dicts. The `_compute_l2_norms` method should handle the simple dict case and skip (or merge) the ubatch case. Since eviction is designed for streaming single-request scenarios, the simple dict case is the primary path.

**Recommendation:** Accept `PerLayerAttnMetadata`, handle dict case directly, skip list case (log warning).

### Eviction Block Invalidation in _update_states

The pre-merge code placed eviction block invalidation right after `self.input_batch.refresh_metadata()`. In v0.19, this same call is at line 1356. The eviction invalidation block should be inserted immediately after line 1356.

The pre-merge code iterates `self.input_batch.block_table.block_tables` (all groups). Per D-06, we target only group 0: `self.input_batch.block_table[0]`. This simplifies the code.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Block freeing through coordinator | Custom block management | `kv_cache_manager.free_blocks()` | Handles multi-type managers, prefix caching |
| L2 norm storage/retrieval | Custom per-request storage | `L2NormCache` singleton | Thread-safe, differential retrieval, already tested |
| KV replacement strategies | New replacement logic | Pre-merge `_replace_kv_caches_{sink,zero,nearby}` | Proven correct, GPU tensor operations |

## Common Pitfalls

### Pitfall 1: EngineCoreOutput Constructor Kwargs Mismatch
**What goes wrong:** v0.19 `EngineCoreOutput` has new fields (`num_external_computed_tokens`, `routed_experts`, `num_nans_in_logits`) not present in pre-merge. Adding `new_l2_norms` kwarg must go alongside these new fields.
**How to avoid:** Copy the existing v0.19 `EngineCoreOutput(...)` call exactly and append `new_l2_norms=new_l2_norms`.

### Pitfall 2: SchedulerOutput Construction Site at Line 914
**What goes wrong:** The `SchedulerOutput` constructor at line 914 does not currently pass `evictable_token_ranges_map`. The field defaults to `None` via dataclass default, so this works silently for non-eviction requests. But the `_process_evictions()` result must be wired in.
**How to avoid:** Add `evictable_token_ranges_map=processed_ranges or None` to the constructor call.

### Pitfall 3: attn_metadata Type in Ubatch Mode
**What goes wrong:** In ubatch mode (DBO), `attn_metadata` is a `list[dict]` not a plain `dict`. Passing this to `_compute_l2_norms` which expects a dict causes a crash.
**How to avoid:** Check type at call site; for list case, either merge dicts or use first ubatch's metadata (L2 norms are per-token across the full batch, not per-ubatch).

### Pitfall 4: _free_request vs _update_states Cleanup Paths
**What goes wrong:** Request cleanup happens in TWO places: `_free_request` (scheduler) and `_update_states` (GPU runner). Missing either causes memory leaks in eviction state dicts.
**How to avoid:** Add cleanup in both: scheduler's `_free_request` cleans `request_eviction_data` and `_l2_norm_last_index`; GPU runner's finished-request loop in `_update_states` cleans `evicted_ranges` and `l2_norm_cache.remove_request()`.

### Pitfall 5: block_table.np vs block_table Access
**What goes wrong:** `BlockTable` has a `block_table` attribute which is a `BufferTensor` with a `.np` property for numpy access. The pre-merge code uses `bt_np = block_table_obj.block_table.np`. This must be preserved exactly.
**How to avoid:** Use `self.input_batch.block_table[0].block_table.np` for numpy array access, `self.input_batch.block_table[0].block_size` for block size.

### Pitfall 6: L2 Norm Import Guard
**What goes wrong:** The pre-merge L2 norm retrieval uses a try/except ImportError guard around `from vllm.v1.attention.l2_norm_cache import get_l2_norm_cache`. This is important for environments where the eviction module isn't available.
**How to avoid:** Preserve the try/except ImportError pattern in the scheduler's `update_from_output`.

### Pitfall 7: update_from_output Has Second EngineCoreOutput Site
**What goes wrong:** There are two `EngineCoreOutput(...)` construction sites in `update_from_output` -- the main one at line 1464 and a secondary one at line 1497 (for failed KV load requests). The secondary site does NOT need `new_l2_norms` (failed requests have no norms), but SCHED-06 requires verifying all sites.
**How to avoid:** Only add `new_l2_norms` to the main EngineCoreOutput at line 1464. The secondary at line 1497 can be left as-is (None default).

## Code Examples

### Scheduler __init__ Eviction State (adapted from pre-merge)
```python
# After self.failed_recving_kv_req_ids (around line 183)

# Thought eviction state
# req_id -> evictable ranges (managed by scheduler)
self.request_eviction_data: dict[str, list[tuple[int, int]]] = {}
# req_id -> last L2 norm index retrieved (differential retrieval)
self._l2_norm_last_index: dict[str, int] = {}
```

### Scheduler _process_evictions (adapted from pre-merge)
```python
def _process_evictions(self) -> dict[str, list[tuple[int, int]]]:
    """Process evictable token ranges and free corresponding physical blocks."""
    if not self.request_eviction_data:
        return {}

    block_size = self.kv_cache_manager.block_size
    if block_size is None:
        return {}

    processed: list[str] = []
    for request_id, ranges in self.request_eviction_data.items():
        blocks_to_free: set[int] = set()
        for start, end in ranges:
            start_block = (start + block_size - 1) // block_size
            end_block = end // block_size
            if start_block < end_block:
                blocks_to_free.update(range(start_block, end_block))

        if blocks_to_free:
            self.kv_cache_manager.free_blocks(request_id, list(blocks_to_free))
        processed.append(request_id)

    processed_ranges: dict[str, list[tuple[int, int]]] = {
        req_id: self.request_eviction_data[req_id]
        for req_id in processed
    }

    for request_id in processed:
        del self.request_eviction_data[request_id]

    return processed_ranges
```

### L2 Norm Retrieval in update_from_output (adapted for v0.19)
```python
# Inside the per-request loop, before EngineCoreOutput construction
new_l2_norms = None
try:
    from vllm.v1.attention.l2_norm_cache import get_l2_norm_cache
except ImportError:
    get_l2_norm_cache = None

if get_l2_norm_cache is not None:
    try:
        if (request.sampling_params is not None
                and request.sampling_params.enable_l2_norms):
            l2_cache = get_l2_norm_cache()
            start_idx = self._l2_norm_last_index.get(req_id, 0)
            norms = l2_cache.get_norms(req_id, start_idx)
            if norms:
                new_l2_norms = norms
                self._l2_norm_last_index[req_id] = start_idx + len(norms)
    except Exception:
        logger.exception(
            "Unexpected error fetching L2 norms for request %s",
            req_id,
        )
```

### GPU Runner Eviction Block Invalidation (adapted for v0.19 block table API)
```python
# After self.input_batch.refresh_metadata()
evicted_ranges = scheduler_output.evictable_token_ranges_map
if evicted_ranges:
    block_table_obj = self.input_batch.block_table[0]  # Group 0 only (D-06)
    block_size = block_table_obj.block_size
    bt_np = block_table_obj.block_table.np

    for req_id, ranges in evicted_ranges.items():
        req_index = self.input_batch.req_id_to_index.get(req_id)
        if req_index is not None:
            # Update CachedRequestState block_ids
            if req_id in self.requests:
                req_state = self.requests[req_id]
                if len(req_state.block_ids) > 0:
                    block_ids_list = req_state.block_ids[0]
                    for start, end in ranges:
                        start_block = (start + block_size - 1) // block_size
                        end_block = end // block_size
                        for block_idx in range(start_block, end_block):
                            if block_idx < len(block_ids_list):
                                block_ids_list[block_idx] = 0

            # Apply replacement strategy and zero block table
            for start, end in ranges:
                start_block = (start + block_size - 1) // block_size
                end_block = end // block_size
                if self.replace_func:
                    # ... (sink/zero/nearby replacement logic, same as pre-merge)
                    pass
                if start_block < end_block:
                    bt_np[req_index, start_block:end_block] = 0
```

## Existing Test Analysis

Tests in `tests/eviction/` that are affected by Phase 3:

| Test File | Type | Phase 3 Impact |
|-----------|------|----------------|
| `test_scheduler_eviction_fix.py` | Unit (monkey-patch) | Uses `Scheduler._process_evictions` directly -- will work once method is added |
| `test_l2_norm_delivery.py` | Source-inspect | Checks `_l2_norm_last_index` in `__init__`, `update_from_output`, `_free_request` source -- will work once code is added |
| `test_block_utils.py` | Unit (pure functions) | No Phase 3 changes needed |
| `test_strategies.py` | Unit (pure functions) | No Phase 3 changes needed |
| `test_orchestrator.py` | Unit (mock-based) | No Phase 3 changes needed |
| `test_segmenter.py` | Unit (pure functions) | No Phase 3 changes needed |
| `test_no_eviction_guard.py` | Source-inspect | May check scheduler/runner source -- verify after porting |
| `test_protocol_extension.py` | Source-inspect | Checks EvictionParams -- Phase 4 scope |
| `test_serving_integration.py` | Source-inspect | Checks serving.py -- Phase 4 scope |
| `test_smoke.py` | Functional | Requires running server -- Phase 4 scope |

**Tests that MUST pass after Phase 3:**
- `test_scheduler_eviction_fix.py` -- directly tests `_process_evictions`
- `test_l2_norm_delivery.py` -- source-inspects scheduler eviction code
- `test_block_utils.py`, `test_strategies.py`, `test_orchestrator.py`, `test_segmenter.py` -- unaffected, should still pass

## Sources

### Primary (HIGH confidence)
- Direct code inspection of current `vllm/v1/core/sched/scheduler.py` (v0.19, 2317 lines)
- Direct code inspection of current `vllm/v1/worker/gpu_model_runner.py` (v0.19, 7059 lines)
- Direct code inspection of pre-merge reference `git show 2ec9a65e84b:vllm/v1/core/sched/scheduler.py` (2039 lines)
- Direct code inspection of pre-merge reference `git show 2ec9a65e84b:vllm/v1/worker/gpu_model_runner.py` (5961 lines)
- `vllm/v1/worker/block_table.py` -- `MultiGroupBlockTable.__getitem__` confirmed at line 313
- `vllm/v1/core/kv_cache_manager.py` -- `free_blocks` at line 430
- `vllm/v1/core/kv_cache_coordinator.py` -- `free_blocks` at line 191
- `vllm/v1/core/sched/output.py` -- `evictable_token_ranges_map` at line 243
- `vllm/v1/engine/__init__.py` -- `EngineCoreOutput.new_l2_norms` at line 172
- Phase 1 audit findings (`.planning/phases/01-groundwork-audit/01-AUDIT-FINDINGS.md`)
- Existing test files in `tests/eviction/`

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH -- all code directly inspected, no external dependencies
- Architecture: HIGH -- insertion points precisely located with line numbers
- Pitfalls: HIGH -- identified through direct comparison of pre-merge vs v0.19 code

**Research date:** 2026-04-08
**Valid until:** 2026-05-08 (stable -- code is already merged, no moving target)
