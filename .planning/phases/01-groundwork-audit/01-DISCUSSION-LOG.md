# Phase 1: Groundwork & Audit - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-08
**Phase:** 01-groundwork-audit
**Areas discussed:** Smoke test design, Audit output format, Bug fix scope

---

## Smoke Test Design

### Q1: What level of integration should the smoke test have?

| Option | Description | Selected |
|--------|-------------|----------|
| Mock-based functional test | Test pipeline with mocked engine client. No GPU. Fast, runs in CI. | |
| Server integration test | Start actual vLLM server, send request, verify full stack. Requires GPU. | ✓ |
| Both layers | Mock-based + server integration. Two test files. | |

**User's choice:** Server integration test
**Notes:** None

### Q2: What model should the smoke test use?

| Option | Description | Selected |
|--------|-------------|----------|
| Tiny reasoning model | Small model that emits <think> tokens | |
| Full DeepSeek-R1 or similar | Production-scale reasoning model | |
| You decide | Claude picks smallest viable model | |

**User's choice:** Other — "Run the model at $HOME/scratch/models/deepseek-8b"
**Notes:** User has a specific local model path for testing

### Q3: What should the smoke test verify as 'passing'?

| Option | Description | Selected |
|--------|-------------|----------|
| L2 norms returned + eviction triggered | Verify L2 norms AND at least one eviction event | |
| Full pipeline observable | L2 norms + eviction event + final SSE chunk with eviction stats | ✓ |
| Minimal: server doesn't crash | Server starts, accepts request, returns response without error | |

**User's choice:** Full pipeline observable
**Notes:** None

### Q4: Should the smoke test run against v0.14 first?

| Option | Description | Selected |
|--------|-------------|----------|
| Run on v0.14 first (Recommended) | Establish baseline before upgrading | ✓ |
| Only run on v0.19 | Write test but only validate upgrade | |

**User's choice:** Run on v0.14 first (Recommended)
**Notes:** None

---

## Audit Output Format

### Q1: Where should audit findings be recorded?

| Option | Description | Selected |
|--------|-------------|----------|
| Dedicated audit doc in .planning/ | Structured findings per item, clean separation | ✓ |
| Inline in CONTEXT.md for Phase 2+ | Fold directly into downstream context | |
| Both | Audit doc + summarize into Phase 2/3 context | |

**User's choice:** Dedicated audit doc in .planning/
**Notes:** None

### Q2: What structure should each finding have?

| Option | Description | Selected |
|--------|-------------|----------|
| Factual summary + code references | What was found, file paths, line numbers, implications. No recommendations. | ✓ |
| Findings + adaptation notes | Facts PLUS notes on what eviction code needs to change | |
| You decide | Claude structures as needed | |

**User's choice:** Factual summary + code references
**Notes:** None

### Q3: How should v0.19 codebase be accessed?

| Option | Description | Selected |
|--------|-------------|----------|
| Git fetch upstream tag | Add upstream remote, fetch v0.19.0 tag. Same repo. | ✓ |
| Separate clone | Clone upstream into separate directory | |
| You decide | Claude picks most practical approach | |

**User's choice:** Git fetch upstream tag
**Notes:** None

---

## Bug Fix Scope

### Q1: How minimal should bug fixes be?

| Option | Description | Selected |
|--------|-------------|----------|
| Strict minimal fix | Fix only the exact crash, nothing else | ✓ |
| Fix + nearby issues | Fix crash AND closely related issues in same file | |
| You decide per case | Claude judges each fix | |

**User's choice:** Strict minimal fix
**Notes:** None

### Q2: How should bug fixes be verified?

| Option | Description | Selected |
|--------|-------------|----------|
| Verify via smoke test | Fixes only "done" when smoke test passes | ✓ |
| Crash-free is sufficient | Verify each fix resolves its specific crash | |

**User's choice:** Verify via smoke test
**Notes:** None

---

## Additional Decisions (from follow-up discussion)

### KV Replacement Strategy
**User's input:** "For the VLLM_KV_REPLACEMENT_STRATEGY, leave the default as None, just leave the garbage values in the blocks which can't be evicted"
**Decision:** VLLM_KV_REPLACEMENT_STRATEGY = None (no replacement)

### CONCERNS.md Tech Debt Items
**User's input:** Fold specific items into Phase 1 scope
**Selected items:**
1. Duplicate import fix (gpu_model_runner.py)
2. Dead FlashAttentionMetadata fields (flash_attn.py)
3. Layer sort bug (_compute_l2_norms)
4. Dead update_norms() method (l2_norm_cache.py)
5. Dual singleton pattern in L2NormCache

---

## Claude's Discretion

- Smoke test file location and naming
- Specific prompt text for smoke test
- Audit doc internal structure
- Order of operations

## Deferred Ideas

- UUID suffix restoration (DEBT-02) — separate effort after upgrade
- Memory pre-allocation optimization — post-upgrade
- update_request_mask overwrite bug — known issue, not Phase 1
- L2NormCache memory leak in multi-process mode — post-upgrade
- No non-streaming eviction path — feature work
- Security: no auth on eviction endpoints — separate concern
