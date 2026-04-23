# Domain Pitfalls: vLLM Fork Upgrade v0.14 → v0.19

**Domain:** vLLM fork upgrade — thought eviction on internal engine APIs
**Researched:** 2026-04-07
**Confidence:** HIGH — derived from direct patch analysis (5,645 lines, 37 files) and existing CONCERNS.md audit

---

## Critical Pitfalls

### Pitfall 1: Source-Inspection Tests Pass While Runtime Is Broken

**What goes wrong:** Nine of nine eviction tests use `inspect.getsource()` or `inspect.signature()` to assert that a variable name or parameter exists in a function body. They do not call the function. After re-applying the patch to v0.19 source, the tests can turn green while the actual eviction pipeline is completely broken.

**Why it happens:** The test suite detects structural omissions, not logic errors. A variable rename causes source-inspect tests to fail, but any logic error that preserves variable names passes silently.

**Consequences:** Fork ships on v0.19 with green CI but eviction produces no norms, delivers wrong norms, or silently discards eviction commands.

**Prevention:**
- Write a functional smoke test BEFORE starting upgrade work. Mock `EngineCore` and drive: `update_request_mask` → `_process_evictions` → `SchedulerOutput` contains ranges → `execute_model` → `EngineCoreOutput.new_l2_norms` non-empty.
- Run the smoke test after re-applying each hunk.
- Treat source-inspect tests as structural regression guards only.

**Warning signs:**
- All 9 tests green but `new_l2_norms` is always `None` in a live request.
- `POST /v1/kv_cache/evict` returns 200 but GPU block table is unchanged.

**Phase:** Phase 1 — write smoke test before merge work begins.

---

### Pitfall 2: `SchedulerOutput` Field Addition Breaks `make_empty()` Construction

**What goes wrong:** `SchedulerOutput.evictable_token_ranges_map` is an optional dict with `default_factory=dict`. Every construction site that omits it silently gets an empty dict. In v0.19, upstream adds new construction sites and refactors existing ones. Missing even one site means the GPU runner always receives an empty map, so KV blocks are never zeroed and eviction silently does nothing.

**Why it happens:** Optional dataclass fields fail silently. The field was added only to `output.py` — every caller of `SchedulerOutput(make_empty=...)` must be audited independently.

**Consequences:** Complete silent eviction failure with no error.

**Prevention:**
- After re-applying `output.py` hunk, run `grep -rn "SchedulerOutput(" vllm/` and `grep -rn "make_empty"`. Verify every construction site passes `evictable_token_ranges_map`.
- Add a deliberate assertion in `_process_evictions`: if `scheduler_output.evictable_token_ranges_map is None` raise immediately.

**Warning signs:**
- `scheduler_output.evictable_token_ranges_map` is always `{}` despite calling `update_request_mask`.
- No "Stored evictable ranges" debug log lines appear.

**Phase:** Phase 2 — check all `SchedulerOutput` construction sites immediately after applying the `output.py` hunk.

---

### Pitfall 3: Block Table Private API Gone in v0.19

**What goes wrong:** `gpu_model_runner._process_evictions` accesses `self.input_batch.block_table.block_tables` (a list of `BlockTableTensor` objects) and then `.block_table.np` (a NumPy view). These are undocumented internals refactored in v0.19 for ubatch scheduling and KV connector support.

**Why it happens:** The patch used the fastest available path rather than the public API. The private path survives v0.14 but is not guaranteed beyond it.

**Consequences:** `AttributeError: 'BlockTable' object has no attribute 'block_tables'` at the start of every forward pass with eviction ranges, crashing the engine process.

**Prevention:**
- Before re-applying the `gpu_model_runner` hunk, run `git diff v0.14.0rc2..v0.19.0 -- vllm/v1/worker/block_table.py` and `git diff v0.14.0rc2..v0.19.0 -- vllm/v1/worker/input_batch.py`.
- If `.block_table.np` is gone, locate the equivalent path through `attn_metadata.block_table` (the tensor exposed by the attention metadata builder).
- Add a defensive `hasattr` check at startup that logs a clear error if the internal path is missing.

**Warning signs:**
- `AttributeError` in `gpu_model_runner.py` on the first request with `eviction_params`.
- `input_batch.block_table` type is different from `BlockTableTensor` in v0.19.

**Phase:** Phase 3 — audit block table API before applying the eviction hunk.

---

### Pitfall 4: `FlashAttentionMetadataBuilder.build()` Signature Changed

**What goes wrong:** The custom patch adds `request_ids` and `compute_l2_norms` as keyword parameters to `build()`. vLLM v0.15+ added `fast_build`; v0.19 adds more parameters for ubatch/chunked-prefill. If v0.19 inserts new parameters before the custom ones, every `builder.build(...)` call raises `TypeError` or silently passes values to wrong parameters.

**Why it happens:** `build()` is called from multiple paths (prefill, decode, CUDAGraph capture). Parameters added at the end are fragile if upstream inserts before them.

**Consequences:** Either a hard `TypeError` on model load, or `compute_l2_norms` silently assigned to a different field.

**Prevention:**
- Run `git diff v0.14.0rc2..v0.19.0 -- vllm/v1/attention/backends/flash_attn.py` before re-applying.
- Since `compute_l2_norms` and `request_ids` are never read inside the kernel (CONCERNS.md confirms this), remove these fields from `FlashAttentionMetadata` entirely — this eliminates the entire conflict zone.

**Warning signs:**
- `TypeError: build() got an unexpected keyword argument` at server startup.
- `compute_l2_norms` is `False` on all decode steps.

**Phase:** Phase 3 — diff `build()` signature before applying; prefer removing the dead fields.

---

### Pitfall 5: `EngineCoreRequestType` Enum Value Collision

**What goes wrong:** The patch adds `UPDATE_MASK = b'\x05'`. If vLLM v0.19 added a new request type using the same byte value, Python's `enum` silently aliases the duplicate — `EngineCoreRequestType.UPDATE_MASK` returns the upstream type, and `EngineCoreProc.handle_client_request` routes eviction updates to the wrong handler.

**Why it happens:** The patch assigns `b'\x05'` without checking what upstream used after v0.14. vLLM adds new request types across releases (DP coordinator, KV connector, utility requests).

**Consequences:** Eviction mask updates silently routed to the wrong handler — either a deserialization error or a silent no-op.

**Prevention:**
- Run `git show v0.19.0:vllm/v1/engine/__init__.py | grep -A20 'EngineCoreRequestType'` and check what byte values are already taken. Pick an unused value.
- Add a test asserting all enum members have distinct values.

**Warning signs:**
- `EngineCoreRequestType.UPDATE_MASK` returns a name other than `'UPDATE_MASK'` in Python REPL after upgrade.
- Eviction requests cause msgspec deserialization errors in `EngineCoreProc`.

**Phase:** Phase 2 — check enum values before applying `__init__.py` hunk.

---

## Moderate Pitfalls

### Pitfall 6: Scheduler `schedule()` Hunk Misplacement

**What goes wrong:** The patch makes 15 separate injections across `Scheduler.__init__`, `schedule()`, `update_from_output()`, and `_free_request()`. The v0.19 `schedule()` added ubatch scheduling and KV-connector callbacks. Fuzzy hunk matching can place `_process_evictions()` after block allocation rather than before it — code compiles, tests pass, but eviction timing is wrong.

**Prevention:**
- Re-apply scheduler hunks manually using semantic anchors (function names, comment strings) rather than line offsets.
- After applying, read the resulting `schedule()` top-to-bottom and verify `_process_evictions()` is called before block allocation for the current batch.
- Verify `_free_request()` still calls both `request_eviction_data.pop()` and `_l2_norm_last_index.pop()`.

**Warning signs:**
- Block double-free warnings in scheduler logs.
- L2 norm indices continue growing after a request finishes.

**Phase:** Phase 2.

---

### Pitfall 7: `free_blocks()` Prefix Cache Interaction

**What goes wrong:** `SingleTypeKVCacheManager.free_blocks()` sets freed slots to `_null_block` but does not update `num_cached_block`. In v0.19, if the KV cache manager was refactored (likely given KV connector changes), this may produce stale prefix cache hits on zeroed blocks.

**Prevention:**
- Run `git diff v0.14.0rc2..v0.19.0 -- vllm/v1/core/single_type_kv_cache_manager.py`.
- After re-applying, test: allocate a request, partially evict blocks 2–4, submit a new request with the same prefix, assert no cache hit on evicted blocks.

**Warning signs:**
- KV cache hit rate unusually high after eviction.
- Garbled output on prefix-sharing requests.

**Phase:** Phase 2.

---

### Pitfall 8: Pydantic v1 `parse_obj` Crash on Eviction Endpoint

**What goes wrong:** `vllm/entrypoints/api_server.py` uses `UpdateMaskRequest.parse_obj(json_request)` — Pydantic v1 API removed in v2. This is a pre-existing crash that will persist through the upgrade and appear as an upgrade regression.

**Prevention:**
- Replace with `UpdateMaskRequest.model_validate(json_request)` in Phase 1 (pre-upgrade cleanup), before the merge. This is a one-line fix.

**Warning signs:**
- `AttributeError: type object 'UpdateMaskRequest' has no attribute 'parse_obj'` on the first eviction request.

**Phase:** Phase 1 — fix before merge.

---

### Pitfall 9: Output Pipeline Hunk Misplacement — Norms Only on Final Token

**What goes wrong:** The hunks in `async_llm.py` and `output_processor.py` that thread `new_l2_norms` from `EngineCoreOutput` to `RequestOutput` may land inside a conditional block (e.g., `if finished:`) in v0.19 due to fuzzy matching. This causes norms to be delivered only on the final token, defeating progressive eviction.

**Prevention:**
- After re-applying, stream a 10-token completion with `enable_l2_norms=True` and assert at least one non-final chunk has `new_l2_norms` set.
- Manually read the diff to confirm norm threading is in the per-token path.

**Warning signs:**
- Orchestrator receives norms only after generation completes.
- `new_l2_norms` is `None` for all intermediate streaming chunks.

**Phase:** Phase 2.

---

### Pitfall 10: `InprocClient` Missing `update_request_mask_async`

**What goes wrong:** The base class raises `NotImplementedError`; `InprocClient` has no override. Any test using in-process mode (e.g., `enforce_eager=True`) that triggers eviction crashes.

**Prevention:**
- Implement `InprocClient.update_request_mask_async` as a direct call to `self.engine_core.scheduler.update_request_mask(request_id, ranges)`. Two-line fix.

**Phase:** Phase 1 — fix before merge.

---

## Minor Pitfalls

### Pitfall 11: `logger.info("sink")` Floods Production Logs

`_replace_kv_caches_sink/zero/nearby` log at INFO level on every boundary block replacement. Change to `logger.debug(...)` in Phase 3.

### Pitfall 12: Duplicate `get_l2_norm_cache` Import

`gpu_model_runner.py` imports `get_l2_norm_cache` twice. Remove the duplicate when re-applying the hunk in Phase 3.

### Pitfall 13: `VLLM_KV_REPLACEMENT_STRATEGY` Registration Pattern May Have Changed

If v0.19 `envs.py` uses a structured registry for environment variables, the custom variable must follow the new pattern. Check `envs.py` diff before applying.

---

## Phase-Specific Warnings Summary

| Phase | Files | Highest-Risk Pitfall | Action Before Apply |
|-------|-------|---------------------|---------------------|
| Phase 1: Pre-upgrade cleanup | `api_server.py`, `core_client.py` | Pydantic v1 crash, InprocClient incomplete | Fix both; write functional smoke test |
| Phase 2: Scheduler + engine IPC | `scheduler.py`, `__init__.py`, `core.py`, `output.py` | Hunk misplacement, enum collision, silent empty map | Semantic re-application; enum audit; grep all SchedulerOutput sites |
| Phase 2: KV cache manager | `single_type_kv_cache_manager.py` | Prefix cache stale hit | Diff file; test partial eviction + prefix request |
| Phase 3: GPU model runner | `gpu_model_runner.py` | Block table private API gone | Diff `block_table.py` and `input_batch.py` first |
| Phase 3: Attention backend | `flash_attn.py` | `build()` signature mismatch | Diff `build()` signature; consider removing unused fields |
| Phase 3: Output pipeline | `async_llm.py`, `output_processor.py` | Norms on final token only | Test intermediate streaming chunks after apply |
| All phases | All test files | Source-inspect masks runtime breakage | Run functional smoke test after every hunk |

---

## Confidence Assessment

| Area | Level | Reason |
|------|-------|--------|
| Block table API breakage | HIGH | CONCERNS.md explicitly flags `block_table.np` as undocumented internal; patch confirms the access pattern |
| Enum collision | HIGH | Patch uses `b'\x05'`; vLLM adds enum values between releases; known pattern from other vLLM forks |
| SchedulerOutput silent failure | HIGH | Optional dict field with `default_factory=dict` — the failure mode is a fundamental Python dataclass property |
| Source-inspect test gap | HIGH | All 9 test files confirmed to use `inspect.getsource()` — zero functional assertions anywhere in the suite |
| `build()` signature | MEDIUM | HIGH probability of change (ubatch added, other params added) but exact v0.19 signature not verified from source |

---

## Open Questions

- What byte values does vLLM v0.19 use in `EngineCoreRequestType`? (Requires checking the v0.19 source file directly.)
- What is the exact shape of `input_batch.block_table` in v0.19? (Requires `git diff v0.14.0rc2..v0.19.0 -- vllm/v1/worker/input_batch.py`.)
- Did v0.19 centralize prefix cache invalidation in a way that requires `free_blocks()` to call a new hook?

---

*Research date: 2026-04-07*
