# Deferred Items - Phase 03 Core Adaptation

## Pre-existing Test Failures (Not Phase 3 Scope)

### 1. test_strategies.py - GlobalStrategy tests (4 failures)
- `test_global_strategy_evicts_highest_norms`
- `test_global_strategy_below_prune_threshold_returns_empty`
- `test_global_strategy_keep_all_returns_empty`
- `test_global_strategy_ranges_are_reasoning_relative`
- **Cause:** Tests pass `prune_after_tokens` keyword to `GlobalStrategy.compute_evictable_ranges()` but the method signature does not accept it. Test/code API mismatch from main branch evolution.

### 2. test_orchestrator.py - Eviction cycle test (1 failure)
- `test_run_eviction_cycle_applies_absolute_offset`
- **Cause:** `update_request_mask` is never called in the mock setup. Pre-existing test/code mismatch.

These failures exist before any Phase 3 changes and are out of scope per deviation rules (only fix issues directly caused by current task's changes).
