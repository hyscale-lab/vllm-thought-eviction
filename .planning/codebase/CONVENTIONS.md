# Coding Conventions

**Analysis Date:** 2026-04-07

## Naming Patterns

**Files:**
- `snake_case.py` for all Python modules: `block_utils.py`, `orchestrator.py`, `segmenter.py`, `strategies.py`
- Test files prefixed with `test_`: `test_strategies.py`, `test_orchestrator.py`, `test_block_utils.py`
- Private module-level constants use `_UPPER_SNAKE_CASE` with leading underscore: `_THINK_START_RE`, `_THINK_END_RE`, `_SEPARATOR_OVERLAP`

**Functions:**
- `snake_case` for all functions and methods: `compute_evictable_ranges`, `merge_overlapping_ranges`, `align_ranges_to_blocks`
- Private helpers prefixed with `_`: `_indices_to_ranges`, `_accumulate`, `_maybe_schedule_cycle`, `_run_eviction_cycle`, `_build_strategy`
- Private helper functions with `_` prefix at module level when not part of a class's public API: `_indices_to_ranges` in `vllm/thought_eviction/strategies.py`

**Variables:**
- `snake_case` throughout: `l2_norms`, `keep_ratio`, `prune_after_tokens`, `eviction_candidates`
- Private instance attributes use leading `_`: `_pending_task`, `_generation_finished`, `_in_think_block`, `_think_start_found`, `_eviction_events`, `_thought_random_scores`

**Classes:**
- `PascalCase` for all classes: `EvictionOrchestrator`, `ThoughtSegmenter`, `ThoughtSegment`, `GlobalStrategy`, `ThoughtMinStrategy`, `ThoughtAvgStrategy`, `RandomStrategy`

**Constants:**
- Module-level class attributes in `UPPER_SNAKE_CASE`: `TARGET_PHRASES`, `_SEPARATOR_OVERLAP` in `vllm/thought_eviction/segmenter.py`
- Private module-level compiled regexes: `_THINK_START_RE`, `_THINK_END_RE` in `vllm/thought_eviction/orchestrator.py`

## Code Style

**Formatter:** ruff-format (via pre-commit hook, ruff v0.14.0)
- Config in `pyproject.toml` under `[tool.ruff.format]`
- `docstring-code-format = true` — code blocks inside docstrings are also formatted

**Linter:** ruff (ruff-check with --fix)
- Rules enabled: pycodestyle (E), Pyflakes (F), pyupgrade (UP), flake8-bugbear (B), flake8-simplify (SIM), isort (I), flake8-logging-format (G)
- Notable ignores: star imports (F403, F405), lambda assignment (E731), zip without strict (B905)
- `vllm/third_party/**` is excluded from all rules

**Type checking:** mypy 1.11.1 with `pydantic.mypy` plugin
- `ignore_missing_imports = true`, `check_untyped_defs = true`, `follow_imports = "silent"`
- Runs for Python 3.10, 3.11, 3.12, 3.13 in CI (manual stage); 3.10 locally on pre-commit

**Spell checking:** typos (pre-commit hook, v1.38.1)

**C++/CUDA:** clang-format with `--style=file` (pre-commit, clang-format v21.1.2)

**Shell scripts:** shellcheck (pre-commit)

## SPDX License Headers

All upstream `vllm/` Python files begin with exactly these two comment lines:
```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
```
Example: `vllm/sampling_params.py`, `vllm/outputs.py`, `vllm/logger.py`.

**The custom `vllm/thought_eviction/` modules do NOT have these headers.** They start directly with a module docstring. This is a divergence from upstream convention. New files added to `vllm/thought_eviction/` do not currently include SPDX headers — but new files added anywhere else in `vllm/` must include them (enforced by the `check-spdx-header` pre-commit hook).

## Import Organization

**Order (enforced by ruff isort rule I):**
1. Standard library imports (stdlib)
2. Third-party imports (numpy, torch, pydantic, etc.)
3. Internal `vllm.*` imports

**Path aliases:** None — all internal imports use full `vllm.*` paths.

**Example from `vllm/thought_eviction/orchestrator.py`:**
```python
import asyncio
import re
import time
from typing import AsyncIterator, Optional

import numpy as np

from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.thought_eviction.segmenter import ThoughtSegmenter
from vllm.thought_eviction.block_utils import (
    merge_overlapping_ranges,
    align_ranges_to_blocks,
    apply_retention_window,
)
```

**Special import rules enforced by pre-commit:**
- `import regex as re` required instead of `import re` (enforced by `enforce-import-regex-instead-of-re` hook)
- Direct `import triton` is forbidden (use `vllm.utils.custom_op` wrappers)
- `pickle`/`cloudpickle` imports are blocked
- Root `vllm/__init__.py` must use lazy imports only (enforced by `check-root-lazy-imports`)

## Docstrings

**Module-level docstrings:** All `vllm/thought_eviction/` modules have multi-line triple-quoted module docstrings explaining purpose, contents, and design constraints.

**Class docstrings:** All public classes have docstrings. Multi-paragraph when needed.

**Method docstrings:** All public methods have docstrings with Args and Returns sections.

**Example pattern from `vllm/thought_eviction/strategies.py`:**
```python
def compute_evictable_ranges(
    self,
    l2_norms: np.ndarray,
    keep_ratio: float,
) -> list[tuple[int, int]]:
    """Compute evictable token ranges using global L2 norm ranking.

    The orchestrator enforces the prune_after_tokens minimum before
    calling this method, so no token-count guard is needed here.

    Args:
        l2_norms: Array of L2 norms for all reasoning tokens.
        keep_ratio: Fraction of tokens to keep (0.0 to 1.0).

    Returns:
        List of reasoning-relative (start, end) ranges to evict.
    """
```

**Test docstrings:** Every test function has a one-line docstring stating what behavior it verifies. Example: `"""Test 1: _accumulate extends accumulated_l2_norms from res.new_l2_norms."""`

## Type Annotations

**Style:** Mix of modern Python 3.10+ builtin generics and `typing` module for complex types.
- Use `list[tuple[int, int]]` (not `List[Tuple[int, int]]`) for builtin generics
- Use `Optional[int]` from `typing` for nullable values (not `int | None` style)
- Return type annotations on all public methods: `-> None`, `-> list[tuple[int, int]]`, etc.
- `from typing import Optional` is explicitly imported in files that use it

**Example:**
```python
from typing import Optional
self.reasoning_start_token_offset: Optional[int] = None
self._pending_task: Optional[asyncio.Task] = None

def _indices_to_ranges(indices: list[int]) -> list[tuple[int, int]]:
```

## Logging

**Framework:** `vllm.logger.init_logger(__name__)` — used in all `vllm/thought_eviction/` modules.

**Pattern:**
```python
from vllm.logger import init_logger
logger = init_logger(__name__)
```

**Log levels used in eviction code:**
- `logger.info(...)` for successful eviction cycle completion
- `logger.error(..., exc_info=True)` for caught exceptions in eviction cycles

**Format:** `%`-style lazy formatting (never f-strings in logger calls), per flake8-logging-format rule G:
```python
logger.info(
    "Eviction cycle %d for %s: %d ranges, %d tokens evicted",
    self.cycle_count, self.request_id, len(aligned), evicted_tokens,
)
```

## Error Handling

**Strategy in `vllm/thought_eviction/orchestrator.py`:**
- Async eviction cycles catch all exceptions to avoid crashing the stream: `except Exception as exc: logger.error(...)`
- `asyncio.CancelledError` is always re-raised: `except asyncio.CancelledError: raise`
- `ValueError` is raised for unknown strategy names in `_build_strategy`
- Early-return guards (not exceptions) used for precondition failures: token threshold, delay intervals, missing state

**Pattern:**
```python
except asyncio.CancelledError:
    raise
except Exception as exc:
    logger.error(
        "Eviction cycle failed for request %s: %s",
        self.request_id, exc, exc_info=True,
    )
```

## Comments

**Section separators in test files:** 79-character dashed lines used to group test cases:
```python
# ---------------------------------------------------------------------------
# Test 1: _accumulate extends accumulated_l2_norms
# ---------------------------------------------------------------------------
```

**Inline comments:** Used sparingly for non-obvious logic. Reference design doc codes inline: `# ENG-06`, `# D-05`, `# Pitfall #6`, `# Guard 1 (ENG-09)`.

**Guard comments:** Each early-return in `_run_eviction_cycle` is preceded by a comment identifying the design spec reference: `# Guard 1 (ENG-09): minimum token threshold`.

## Function Design

**Size:** Functions are focused on a single responsibility. The longest method is `_run_eviction_cycle` (~130 lines) — it is structured with clearly labeled guard sections and a single linear processing pipeline.

**Parameters:** Keyword-only parameters for boolean flags and optional overrides. Factory helpers (`make_eviction_params`, `make_orchestrator`) accept `**overrides` dicts for test flexibility.

**Return values:** Always explicitly typed. Pure functions return new objects (never mutate input). Methods that mutate state return `None`.

## Module Design

**Exports:** No `__all__` declarations in `vllm/thought_eviction/` modules — all public names are importable directly.

**Barrel files:** `vllm/thought_eviction/__init__.py` is empty — callers import from specific submodules, e.g., `from vllm.thought_eviction.strategies import GlobalStrategy`.

**Singleton pattern:** `vllm/v1/attention/l2_norm_cache.py` exposes a `get_l2_norm_cache()` factory function returning the process-level singleton instance.

---

*Convention analysis: 2026-04-07*
