# Phase 1: Groundwork & Audit - Context

**Gathered:** 2026-04-08
**Status:** Ready for planning

<domain>
## Phase Boundary

Fix pre-existing bugs in the current v0.14 eviction code, clean up dead/broken code that would complicate the upgrade, write a functional server-level smoke test, and resolve all structural unknowns about v0.19 before any merge work begins.

</domain>

<decisions>
## Implementation Decisions

### Smoke Test Design
- **D-01:** The smoke test is a server integration test — start the actual vLLM server, send a chat completion request with `eviction_params`, and verify the full pipeline observable result.
- **D-02:** Use the model at `$HOME/scratch/models/deepseek-8b` for inference.
- **D-03:** Passing criteria: L2 norms returned in the response, at least one eviction event occurs, AND the final SSE chunk contains eviction statistics (evicted token count, strategy used).
- **D-04:** Run the smoke test on the current v0.14 codebase first to establish a baseline before any upgrade work. If it fails on v0.14, the bug is pre-existing.

### Audit Output Format
- **D-05:** Audit findings (AUDIT-01 through AUDIT-05) go into a dedicated doc: `.planning/phases/01-groundwork-audit/01-AUDIT-FINDINGS.md`.
- **D-06:** Each finding uses factual summary + code references format: what was found, exact file paths and line numbers in v0.19, and implications for eviction code. No recommendations — just facts.
- **D-07:** Access v0.19 codebase by adding upstream vLLM as a git remote and fetching the v0.19.0 tag (commit 2a69949). Audit by checking out or diffing against that ref.

### Bug Fix Scope
- **D-08:** Bug fixes (CLEAN-01, CLEAN-02) must be strict minimal fixes — fix only the exact crash, don't touch anything else in those files.
- **D-09:** Bug fixes are verified via the smoke test — fixes are only "done" when the smoke test passes.

### KV Replacement Strategy
- **D-10:** `VLLM_KV_REPLACEMENT_STRATEGY` stays as `None` (default). Leave garbage values in non-evictable boundary block slots. No replacement strategy needed for the upgrade.

### Folded Cleanup Items (from CONCERNS.md)
- **D-11:** Remove duplicate `get_l2_norm_cache` import in `vllm/v1/worker/gpu_model_runner.py` (line 171 duplicate of line 129).
- **D-12:** Remove unused `compute_l2_norms` and `request_ids` fields from `FlashAttentionMetadata` in `vllm/v1/attention/backends/flash_attn.py`, and stop populating them in `extra_attn_metadata_args`. These are dead fields from an earlier design.
- **D-13:** Fix `_compute_l2_norms` layer sorting bug — replace string-sorted `sorted(attn_metadata_dict.keys())` with numeric sort to match `self.kv_caches` physical order.
- **D-14:** Delete dead `update_norms()` method in `vllm/v1/attention/l2_norm_cache.py` (lines 216-238) — wrong arity, never called.
- **D-15:** Fix dual singleton pattern in `L2NormCache` — keep only the module-level `_l2_norm_cache` global and `get_l2_norm_cache()` factory; remove the `__new__`-based `_instance` singleton.

### Claude's Discretion
- Smoke test file location and naming within `tests/eviction/`
- Specific prompt text used for the smoke test chat completion request
- Audit doc internal structure and formatting
- Order of operations (bugs first vs audit first vs smoke test first)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Eviction Module (current v0.14 code)
- `vllm/thought_eviction/orchestrator.py` — EvictionOrchestrator, stream wrapping, eviction cycle logic
- `vllm/thought_eviction/segmenter.py` — ThoughtSegmenter, ThoughtSegment dataclass
- `vllm/thought_eviction/strategies.py` — All four strategy classes
- `vllm/thought_eviction/block_utils.py` — Range merge, block-align, retention window utilities

### L2 Norm Pipeline
- `vllm/v1/attention/l2_norm_cache.py` — L2NormCache singleton (dual singleton bug target)
- `vllm/v1/worker/gpu_model_runner.py` — `_compute_l2_norms`, block invalidation, duplicate import target
- `vllm/v1/attention/backends/flash_attn.py` — Dead FlashAttentionMetadata fields target

### Engine/Scheduler
- `vllm/v1/core/sched/scheduler.py` — update_request_mask, _process_evictions
- `vllm/v1/core/sched/output.py` — SchedulerOutput.evictable_token_ranges_map
- `vllm/v1/engine/core_client.py` — InprocClient missing method (CLEAN-02 target)
- `vllm/v1/engine/__init__.py` — EngineCoreOutput.new_l2_norms, EngineCoreRequestType

### Serving Layer
- `vllm/entrypoints/api_server.py` — Pydantic parse_obj crash (CLEAN-01 target)
- `vllm/entrypoints/openai/chat_completion/serving.py` — Eviction stream wrapping
- `vllm/entrypoints/openai/chat_completion/protocol.py` — EvictionParams, ChatCompletionRequest

### Codebase Analysis
- `.planning/codebase/CONCERNS.md` — Full list of known bugs, tech debt, and fragile areas
- `.planning/codebase/ARCHITECTURE.md` — Layer architecture and data flow
- `.planning/codebase/STRUCTURE.md` — File layout and modification map

### Existing Tests
- `tests/eviction/` — All 9 existing eviction test files (source-inspect based)

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- Existing `tests/eviction/` test infrastructure can inform smoke test structure
- `eviction_changes.patch` contains the complete diff of all fork changes — useful for identifying all modified files

### Established Patterns
- vLLM uses pytest as test runner (configured in `pyproject.toml`)
- Eviction tests currently use `inspect.getsource()` pattern — smoke test should NOT follow this pattern (use real server integration)
- `import regex as re` required instead of `import re` (enforced by pre-commit)

### Integration Points
- Smoke test connects to the server via HTTP (FastAPI endpoints)
- Audit connects to upstream git remote for v0.19 source access

</code_context>

<specifics>
## Specific Ideas

- Model for smoke test: `$HOME/scratch/models/deepseek-8b`
- KV replacement strategy: explicitly None — no fill needed for garbage boundary slots

</specifics>

<deferred>
## Deferred Ideas

- UUID suffix restoration (DEBT-02) — separate effort after upgrade
- Memory pre-allocation optimization (MAX_SEQ_LEN=100000) — post-upgrade improvement
- `update_request_mask` overwrite bug — known issue, not in Phase 1 scope
- L2NormCache memory leak in multi-process mode (orchestrator cleanup no-op) — post-upgrade
- No non-streaming eviction path — feature work, not upgrade scope
- Security: no auth on eviction endpoints — separate concern from upgrade

</deferred>

---

*Phase: 01-groundwork-audit*
*Context gathered: 2026-04-08*
