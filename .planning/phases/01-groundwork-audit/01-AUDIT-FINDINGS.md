# Phase 1: Audit Findings

**Audited:** 2026-04-08
**v0.19.0 ref:** commit 2a69949 (tag `v0.19.0`)
**Verified against:** `git show v0.19.0:<path>` commands on all source files

---

## AUDIT-01: EngineCoreRequestType Byte Values

**File:** `vllm/v1/engine/__init__.py`, lines 217-230

**Byte value assignments in v0.19.0:**

| Value | Name | Status |
|-------|------|--------|
| `b"\x00"` | `ADD` | Same as v0.14 |
| `b"\x01"` | `ABORT` | Same as v0.14 |
| `b"\x02"` | `START_DP_WAVE` | New in v0.19 (was `PROFILE` in v0.14) |
| `b"\x03"` | `UTILITY` | New in v0.19 (replaces several individual request types) |
| `b"\x04"` | `EXECUTOR_FAILED` | New sentinel for engine core process |
| `b"\x05"` | `WAKEUP` | New sentinel to wake up `input_queue.get()` during shutdown |

**Collision confirmed:** The fork's `UPDATE_MASK = b'\x05'` (used for eviction mask IPC) collides directly with upstream's `WAKEUP = b"\x05"`. Both are used in the ZMQ socket protocol between engine core and clients.

**Next free byte value:** `b"\x06"` is the first unused value in v0.19.0. The fork's `UPDATE_MASK` must be reassigned to `b"\x06"` or higher.

**Implication for eviction code:** If the fork's `UPDATE_MASK` remains at `b'\x05'`, the engine core process would misinterpret eviction mask updates as shutdown wakeup signals (or vice versa), causing either silent eviction failures or spurious shutdown behavior.

---

## AUDIT-02: SamplingParams Field Differences (v0.14 -> v0.19)

**File:** `vllm/sampling_params.py`, lines 156-298 (v0.19.0)

**Class declaration changed:** v0.19 adds `dict=True` keyword argument to the `msgspec.Struct` base:
```python
class SamplingParams(
    PydanticMsgspecMixin,
    msgspec.Struct,
    omit_defaults=True,
    dict=True,          # NEW in v0.19
):
```

### Fields Added in v0.19

| Field | Type | Default | Line | Notes |
|-------|------|---------|------|-------|
| `_eos_token_id` | `int \| None` | `None` | 263 | Private, set in post_init |
| `logit_bias` | `dict[int, float] \| None` | `None` | 269 | Logits processor construction |
| `allowed_token_ids` | `list[int] \| None` | `None` | 272 | Logits processor construction |
| `extra_args` | `dict[str, Any] \| None` | `None` | 275 | Custom sampling implementations |
| `bad_words` | `list[str] \| None` | `None` | 281 | Bad words filtering |
| `_bad_words_token_ids` | `list[list[int]] \| None` | `None` | 285 | Private, set in post_init |
| `skip_reading_prefix_cache` | `bool \| None` | `None` | 287 | Prefix cache control |
| `thinking_token_budget` | `int \| None` | `None` | 288 | Thinking token limit |
| `repetition_detection` | `RepetitionDetectionParams \| None` | `None` | 291 | N-gram repetition detection |

### Fields Removed in v0.19 (Present in v0.14)

| Field | Type | v0.14 Line | Notes |
|-------|------|-----------|-------|
| `logits_processors` | `Any \| None` | 210 | Removed; logits processors now constructed from structured fields |
| `truncate_prompt_tokens` | `Annotated[int, msgspec.Meta(ge=-1)] \| None` | 215 | Removed entirely |

### Fields Changed

- `output_text_buffer_length` moved from line 241 (v0.14) to line 262 (v0.19), and `_eos_token_id` was inserted before `_all_stop_token_ids`
- `n` field docstring expanded with `VLLM_MAX_N_SEQUENCES` documentation (v0.19 line 170-180)
- `flat_logprobs` field added between `prompt_logprobs` and `detokenize` (v0.19 line 235)
- `skip_clone` docstring expanded (v0.19 line 253-258)

### Eviction Field Impact

The fork currently has `enable_l2_norms` and `l2_norm_layers` inserted at lines 226-237 (after `skip_clone`, before `output_text_buffer_length`). In v0.19, these fields must be re-appended AFTER `repetition_detection` (the last v0.19 upstream field at line 291) and BEFORE the internal post_init fields (`output_text_buffer_length`, `_eos_token_id`, `_all_stop_token_ids`).

Because `SamplingParams` uses `omit_defaults=True` with `array_like` semantics in msgspec, positional field ordering is byte-significant for IPC serialization. Incorrect positioning will cause silent data corruption.

---

## AUDIT-03: InputBatch Block Table API (v0.19)

**File:** `vllm/v1/worker/block_table.py`, lines 17-222 (`BlockTable`), lines 223-316 (`MultiGroupBlockTable`)

### MultiGroupBlockTable Structure

`MultiGroupBlockTable` (line 223) wraps multiple `BlockTable` instances, one per KV cache group. This replaces direct block table access patterns from earlier versions.

**Constructor** (lines 226-274):
- Parameters: `max_num_reqs`, `max_model_len`, `max_num_batched_tokens`, `pin_memory`, `device`, `block_sizes`, `kernel_block_sizes`, `max_num_blocks`, `cp_kv_cache_interleave_size`
- Creates one `BlockTable` per entry in `block_sizes`/`kernel_block_sizes` lists
- Stores them in `self.block_tables` (line 260)

**Access patterns:**
- `MultiGroupBlockTable[idx]` returns the `BlockTable` for the i-th KV cache group (line 314: `__getitem__`)
- Delegate methods iterate all block tables: `append_row`, `add_row`, `clear_row`, `move_row`, `swap_row`, `compute_slot_mapping`, `commit_block_table`, `clear`

### Individual BlockTable Structure

`BlockTable` (line 17) manages a single KV cache group:
- `self.block_table` is a `CpuGpuBuffer` of shape `(max_num_reqs, max_num_blocks_per_req)`, dtype `int32` (line 78)
- **NumPy access:** `block_table.block_table.np` returns the CPU-side numpy array (via `CpuGpuBuffer`)
- **GPU tensor access:** `block_table.block_table.gpu` returns the device tensor
- **Convenience:** `num_blocks_per_row` is a numpy array tracking block counts per request (line 79)
- **New feature:** Hybrid blocks support (line 52-68) where `kernel_block_size != block_size` (e.g., 32-token allocation blocks with 16-token kernel blocks)

### Incompatibility with Fork

The fork's eviction code accesses `.block_tables` (plural) as a list-like object directly. In v0.19, `.block_tables` is the internal list attribute of `MultiGroupBlockTable` (line 260), not the top-level `InputBatch` attribute. The fork must access block table data through `MultiGroupBlockTable[group_idx].block_table.np` for numpy arrays, or iterate via the delegate methods.

---

## AUDIT-04: schedule() Structure (v0.19)

**File:** `vllm/v1/core/sched/scheduler.py`

### Method Signature and Location

- `def schedule(self) -> SchedulerOutput` at line 348
- Returns a `SchedulerOutput` dataclass (defined in `vllm/v1/core/sched/output.py`, line 179)

### Two Main Loops

1. **RUNNING requests loop** (lines 383-530): Iterates `self.running` with a `while` loop and `token_budget` guard. Handles spec decode tokens, preemption, KV connector states, encoder scheduling. Appends to `scheduled_running_reqs`.

2. **WAITING requests loop** (lines 563-857): Iterates `self.waiting` and `self.skipped_waiting` queues. Guarded by `not preempted_reqs and self._pause_state == PauseState.UNPAUSED`. Handles new request admission, KV cache allocation, encoder budget, LoRA limits.

### SchedulerOutput Construction

`SchedulerOutput` constructed at line 914 with these fields:

| Field | Type |
|-------|------|
| `scheduled_new_reqs` | `list[NewRequestData]` |
| `scheduled_cached_reqs` | `CachedRequestData` |
| `num_scheduled_tokens` | `dict[str, int]` |
| `total_num_scheduled_tokens` | `int` |
| `scheduled_spec_decode_tokens` | `dict[str, list[int]]` |
| `scheduled_encoder_inputs` | `dict[str, list[int]]` |
| `num_common_prefix_blocks` | `list[int]` |
| `preempted_req_ids` | `set[str]` |
| `finished_req_ids` | `set[str]` |
| `free_encoder_mm_hashes` | `list[str]` |
| `new_block_ids_to_zero` | `list[int] \| None` |

Additional optional fields set after construction:
- `kv_connector_metadata` (line 939)
- `ec_connector_metadata` (line 945)
- `has_structured_output_requests` (line 221 in output.py)
- `pending_structured_output_tokens` (line 225 in output.py)
- `num_invalid_spec_tokens` (line 228 in output.py)

**No `evictable_token_ranges_map` field exists in v0.19 upstream.** This must be added to `SchedulerOutput` during the upgrade.

### Key Method Locations

| Method | Line | Purpose |
|--------|------|---------|
| `schedule()` | 348 | Main scheduling entry point |
| `_update_after_schedule()` | 978 | Post-schedule bookkeeping (computed tokens, spec decode) |
| `update_from_output()` | 1302 | Process model execution results |
| `_free_request()` | 1818 | Clean up finished/aborted requests |
| `_preempt_request()` | 965 | Move request from running back to waiting |

### Hook Insertion Points for Eviction

- `update_request_mask` (fork method): Must be added as a new method on `Scheduler`. Called from engine core IPC handler.
- `_process_evictions` (fork method): Must run before `SchedulerOutput` construction (before line 914) to inject `evictable_token_ranges_map` into the output.
- `_free_request` (line 1818): Eviction cleanup (removing pending eviction ranges) should be added here.

### New Concepts in v0.19 schedule()

- `PauseState` enum controls whether WAITING queue is processed (line 564)
- `use_v2_model_runner` flag affects request data construction
- KV connector integration (`self.connector`) for disaggregated prefill
- EC connector integration (`self.ec_connector`) for elastic compute
- Encoder cache management (`self.encoder_cache_manager`)
- Spec decode scheduling integrated into the main loop

---

## AUDIT-05: EngineCoreClient Subclasses (v0.19)

**File:** `vllm/v1/engine/core_client.py`

### All Classes

| Class | Line | IPC Type | Purpose |
|-------|------|----------|---------|
| `EngineCoreClient` (ABC) | 69 | N/A | Abstract base class defining method stubs for all clients |
| `InprocClient` | 274 | Direct (no IPC) | In-process client for `LLMEngine`; directly holds `EngineCore` reference |
| `BackgroundResources` | 368 | N/A | Helper class managing background ZMQ resources (not a client) |
| `ElasticScalingCache` | 454 | N/A | Helper class for elastic scaling state (not a client) |
| `MPClient` | 460 | ZMQ (base) | Base multi-process client; sets up ZMQ sockets and encoder/decoder |
| `SyncMPClient` | 703 | ZMQ (sync) | Synchronous ZMQ client for blocking `LLM` usage |
| `AsyncMPClient` | 874 | ZMQ (async) | Async ZMQ client; main production path for `AsyncLLM` |
| `DPAsyncMPClient` | 1124 | ZMQ (async, multi-engine) | Data-parallel async client managing multiple engine cores |
| `DPLBAsyncMPClient` | 1304 | ZMQ (async, load-balanced) | Data-parallel with load balancing; **NEW in v0.19** |

### Client Hierarchy

```
EngineCoreClient (ABC, line 69)
  +-- InprocClient (line 274)
  +-- MPClient (line 460)
        +-- SyncMPClient (line 703)
        +-- AsyncMPClient (line 874)
              +-- DPAsyncMPClient (line 1124)
                    +-- DPLBAsyncMPClient (line 1304)
```

### IPC Implications for Eviction Methods

**Direct access (InprocClient):** Can call `self.engine_core.scheduler.update_request_mask()` directly. No serialization needed.

**ZMQ-based (MPClient and subclasses):** Must serialize eviction commands as `EngineCoreRequestType` messages over ZMQ sockets. This is where the byte value collision (AUDIT-01) is critical -- `UPDATE_MASK` must not collide with `WAKEUP`.

**New classes requiring eviction support:**
- `DPLBAsyncMPClient` (line 1304) is entirely new in v0.19 and has no eviction method stubs. It inherits from `DPAsyncMPClient` which also needs eviction methods.
- `BackgroundResources` (line 368) and `ElasticScalingCache` (line 454) are helper classes, not clients -- they do not need eviction methods.

**All 5 actual client subclasses** (`InprocClient`, `SyncMPClient`, `AsyncMPClient`, `DPAsyncMPClient`, `DPLBAsyncMPClient`) need `update_request_mask_async` and `get_request_l2_norms_async` implementations. `MPClient` provides the ZMQ infrastructure but the actual method implementations go on the sync/async subclasses.
