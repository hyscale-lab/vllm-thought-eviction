# Testing Patterns

**Analysis Date:** 2026-04-07

## Test Framework

**Runner:** pytest (version inferred from `__pycache__` filenames: `cpython-312-pytest-9.0.2`)
- Config: `pyproject.toml` under `[tool.pytest.ini_options]`
- Markers defined: `slow_test`, `skip_global_cleanup`, `core_model`, `hybrid_model`, `cpu_model`, `cpu_test`, `split`, `distributed`, `skip_v1`, `optional`

**Assertion Library:** pytest's built-in `assert` (primary) and `unittest.TestCase` assertion methods (in some eviction tests)

**Coverage:** `codecov.yml` present but no coverage thresholds enforced in config.

**Run Commands:**
```bash
pytest tests/eviction/              # Run all eviction tests
pytest tests/eviction/test_strategies.py  # Run a specific file
pytest tests/eviction/ -v           # Verbose output
pytest tests/ -m "not slow_test"    # Skip slow tests
```

## Test File Organization

**Location:** Separate `tests/` directory at project root, not co-located with source.

**Eviction-specific directory:** `tests/eviction/` contains all custom thought-eviction tests.

**Naming:** `test_<module_or_feature>.py`, matching the module under test:
- `tests/eviction/test_strategies.py` → `vllm/thought_eviction/strategies.py`
- `tests/eviction/test_segmenter.py` → `vllm/thought_eviction/segmenter.py`
- `tests/eviction/test_orchestrator.py` → `vllm/thought_eviction/orchestrator.py`
- `tests/eviction/test_block_utils.py` → `vllm/thought_eviction/block_utils.py`
- `tests/eviction/test_l2_norm_delivery.py` → IPC pipeline (scheduler → output_processor → RequestOutput)
- `tests/eviction/test_serving_integration.py` → `vllm/entrypoints/openai/chat_completion/serving.py`
- `tests/eviction/test_protocol_extension.py` → `vllm/entrypoints/openai/chat_completion/protocol.py`
- `tests/eviction/test_no_eviction_guard.py` → SamplingParams guard + L2NormCache filter
- `tests/eviction/test_scheduler_eviction_fix.py` → `vllm/v1/core/sched/scheduler.py` eviction data clearing

**Module marker:** `tests/eviction/__init__.py` present (empty), making it a proper package.

**Structure:**
```
tests/eviction/
├── __init__.py
├── test_block_utils.py          # Pure function tests
├── test_l2_norm_delivery.py     # IPC pipeline tests (unittest.TestCase)
├── test_no_eviction_guard.py    # Guard/filter tests (unittest.TestCase)
├── test_orchestrator.py         # Async orchestrator tests (pytest)
├── test_protocol_extension.py   # Protocol model validation (pytest)
├── test_scheduler_eviction_fix.py  # Scheduler mutation tests (pytest)
├── test_segmenter.py            # Segmenter tests (pytest with classes)
└── test_strategies.py           # Strategy unit tests (pytest)
```

## Test Structure

**Two co-existing styles in eviction tests:**

**Style 1 — pytest flat functions** (used in `test_strategies.py`, `test_orchestrator.py`, `test_block_utils.py`, `test_protocol_extension.py`, `test_scheduler_eviction_fix.py`):
```python
"""
Unit tests for vllm.thought_eviction.strategies.

Coverage:
- STRAT-01: GlobalStrategy sorts by L2 norm and evicts highest-norm tokens
- STRAT-02: ThoughtMinStrategy sorts thoughts by min norm and evicts fraction
"""

import numpy as np
import pytest

from vllm.thought_eviction.strategies import GlobalStrategy, _indices_to_ranges


# ---------------------------------------------------------------------------
# GlobalStrategy — STRAT-01
# ---------------------------------------------------------------------------

def test_global_strategy_evicts_highest_norms():
    """Test 1: 10 tokens, keep_ratio=0.5 — lowest 5 norms kept, rest evicted."""
    strategy = GlobalStrategy()
    norms = np.array([0.5, 0.1, 0.9, 0.3, 0.7])
    ranges = strategy.compute_evictable_ranges(l2_norms=norms, keep_ratio=0.5)
    assert ranges == [...]
```

**Style 2 — pytest classes** (used in `test_segmenter.py`, `test_block_utils.py`):
```python
class TestExtractReasoningSpan:
    """SEG-01: extract_reasoning_span detects <think>...</think> spans."""

    def test_extract_reasoning_span_returns_content_offsets(self, mock_tokenizer):
        """extract_reasoning_span returns character offsets of content inside tags."""
        segmenter = ThoughtSegmenter(mock_tokenizer)
        ...
        assert span is not None
```

**Style 3 — unittest.TestCase** (used in `test_l2_norm_delivery.py`, `test_no_eviction_guard.py`):
```python
class TestEngineCoreOutputField(unittest.TestCase):
    """Test 1 & 2: EngineCoreOutput has new_l2_norms field with None default."""

    def test_engine_core_output_default_none(self):
        """Test 1: EngineCoreOutput without new_l2_norms defaults to None."""
        from vllm.v1.engine import EngineCoreOutput
        e = EngineCoreOutput(request_id="test", new_token_ids=[1])
        self.assertIsNone(e.new_l2_norms)
```

**Module docstring convention:** Every test file starts with a triple-quoted module docstring listing the spec codes covered:
```python
"""
Unit tests for vllm.thought_eviction.strategies.

Coverage:
- STRAT-01: GlobalStrategy sorts by L2 norm and evicts highest-norm tokens
- STRAT-02: ThoughtMinStrategy sorts thoughts by min norm and evicts fraction
"""
```

**Section separators:** Tests within a file are grouped by spec code using 75-character dashed comment blocks:
```python
# ---------------------------------------------------------------------------
# GlobalStrategy — STRAT-01
# ---------------------------------------------------------------------------
```

## Mocking

**Framework:** `unittest.mock` — `MagicMock`, `AsyncMock`, `patch`, `PropertyMock`

**Tokenizer mock pattern** (reused across `test_segmenter.py` and `test_orchestrator.py`):
```python
def make_mock_tokenizer():
    """Build a mock tokenizer returning ~1 token per 5 characters."""
    tokenizer = MagicMock()

    def tokenizer_call(text, add_special_tokens=False, return_offsets_mapping=False):
        token_ids = []
        offsets = []
        pos = 0
        while pos < len(text):
            end = min(pos + 5, len(text))
            token_ids.append(pos)
            offsets.append((pos, end))
            pos = end
        result = {"input_ids": token_ids}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result

    tokenizer.side_effect = tokenizer_call
    return tokenizer
```

**Fixture variant** in `test_segmenter.py`:
```python
@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer returning ~1 token per 5 characters."""
    tokenizer = MagicMock()
    # ... same implementation
    return tokenizer
```

**AsyncMock for engine client:**
```python
engine_client = AsyncMock()
engine_client.update_request_mask = AsyncMock(return_value=True)
```

**patch for asyncio.create_task:**
```python
with patch("asyncio.create_task", return_value=fake_task) as mock_create:
    orc._maybe_schedule_cycle()
    mock_create.assert_called_once()
```

**What to mock:**
- Tokenizer: always mocked in eviction tests (deterministic 5-chars-per-token behavior)
- Engine client: always mocked as `AsyncMock` with `update_request_mask`
- `asyncio.create_task`: mocked when verifying scheduling behavior
- Heavy scheduler dependencies: `object.__new__(Scheduler)` monkey-patching to bypass CUDA imports

**What NOT to mock:**
- Pure functions (`merge_overlapping_ranges`, `align_ranges_to_blocks`, `apply_retention_window`, `_indices_to_ranges`) — tested with real inputs
- `EvictionParams` (pydantic model) — instantiated directly in tests
- `ThoughtSegment` dataclass — instantiated directly via helper factories

## Fixtures and Factories

**pytest fixtures:** Defined in test files (not a shared conftest) for eviction tests:
- `mock_tokenizer` in `tests/eviction/test_segmenter.py`

**Factory helper functions** (plain functions, not fixtures, reused within the same test module):

In `tests/eviction/test_strategies.py`:
```python
def make_thought(text, start_char, end_char, start_tok, end_tok,
                 min_norm=float("inf"), avg_norm=float("inf"),
                 l2_norms=None) -> ThoughtSegment:
    """Build a ThoughtSegment with known field values for testing."""
    ...

def make_thought_no_l2(text, start_char, end_char, start_tok, end_tok) -> ThoughtSegment:
    """Build a ThoughtSegment without L2 norms (for testing filtering)."""
    ...
```

In `tests/eviction/test_orchestrator.py`:
```python
def make_completion_output(text="", token_ids=None):
    """Build a minimal CompletionOutput for testing."""
    ...

def make_request_output(text="", new_l2_norms=None, prompt_token_ids=None,
                        finished=False, token_ids=None):
    """Build a minimal RequestOutput for testing."""
    ...

def make_eviction_params(**overrides):
    """Build EvictionParams with safe test defaults."""
    defaults = dict(strategy="thought_min", keep_ratio=0.5, ...)
    defaults.update(overrides)
    return EvictionParams(**defaults)

def make_orchestrator(params=None, engine_client=None, tokenizer=None,
                      request_id="req-test", block_size=16):
    """Build an EvictionOrchestrator with test defaults."""
    ...
```

In `tests/eviction/test_scheduler_eviction_fix.py`:
```python
def make_scheduler_stub(block_size=16):
    """Return a minimal object mimicking the scheduler eviction interface."""
    sched = object.__new__(Scheduler)
    sched.request_eviction_data = {}
    sched.kv_cache_manager = mock_kv
    return sched
```

**Location:** All factory helpers are defined at the top of the test module they serve — no shared fixture file within `tests/eviction/`.

**Global conftest:** `tests/conftest.py` provides shared fixtures for the broader vLLM test suite (LLM instances, model fixtures, etc.) but is not used by the eviction-specific tests.

## Async Testing

**Pattern:** `asyncio.run()` wrapping synchronous test functions — no `pytest-asyncio`:
```python
def test_run_eviction_cycle_early_return_prune_after_tokens():
    """Test 7: ..."""
    orc = make_orchestrator(...)
    asyncio.run(orc._run_eviction_cycle())
    engine_client.update_request_mask.assert_not_called()
```

**Async generator testing:**
```python
def test_wrap_stream_yields_all_items_unchanged():
    async def mock_generator():
        for o in outputs:
            yield o

    async def run():
        collected = []
        async for res in orc.wrap_stream(mock_generator()):
            collected.append(res)
        return collected

    collected = asyncio.run(run())
    assert len(collected) == 3
```

## Source Inspection Tests

A distinctive pattern in the eviction test suite: tests that verify integration wiring by inspecting source code directly using `inspect.getsource()` and `inspect.signature()`.

Used in `tests/eviction/test_serving_integration.py` and `tests/eviction/test_l2_norm_delivery.py`:
```python
def test_orchestrator_activation_code_present():
    """Verify the conditional activation pattern exists in serving.py source."""
    from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
    source = inspect.getsource(OpenAIServingChat.create_chat_completion)
    assert 'request.eviction_params is not None' in source
    assert 'EvictionOrchestrator(' in source
```

Also used: `pathlib.Path(...).read_text()` for files where `inspect.getsource` can't reach (raw path reading in `test_no_eviction_guard.py`):
```python
src = pathlib.Path(
    "/export/home2/broc/vllm-thought-eviction/vllm/v1/core/sched/scheduler.py"
).read_text()
self.assertIn("sampling_params.enable_l2_norms", src)
```
**Note:** These hardcoded absolute paths are environment-specific and will break outside the original development machine.

## Validation Error Testing

**pydantic ValidationError pattern** in `tests/eviction/test_protocol_extension.py`:
```python
def test_invalid_strategy_rejected() -> None:
    """EvictionParams with an unknown strategy raises ValidationError."""
    with pytest.raises(ValidationError):
        EvictionParams(strategy="invalid")
```

## Coverage

**Requirements:** No enforced minimum coverage thresholds found in `codecov.yml` or `pyproject.toml`.

**What is covered by `tests/eviction/`:**
- `vllm/thought_eviction/strategies.py` — GlobalStrategy, ThoughtMinStrategy, ThoughtAvgStrategy, RandomStrategy, `_indices_to_ranges`
- `vllm/thought_eviction/segmenter.py` — ThoughtSegmenter, ThoughtSegment, incremental tokenization, chunk-boundary detection, sub-threshold merging
- `vllm/thought_eviction/block_utils.py` — all three pure functions and their critical ordering
- `vllm/thought_eviction/orchestrator.py` — accumulate, scheduling, eviction cycle guards, wrap_stream, per-request isolation
- `vllm/entrypoints/openai/chat_completion/protocol.py` — EvictionParams model validation
- `vllm/entrypoints/openai/chat_completion/serving.py` — orchestrator activation wiring (via source inspection)
- `vllm/v1/engine/__init__.py` — EngineCoreOutput.new_l2_norms field
- `vllm/outputs.py` — RequestOutput.new_l2_norms field
- `vllm/sampling_params.py` — SamplingParams.enable_l2_norms field
- `vllm/v1/core/sched/scheduler.py` — L2 norm index tracking and cleanup (via source inspection + monkey-patch)
- `vllm/v1/attention/l2_norm_cache.py` — update_norms_batch eviction filter

**Not covered (no GPU/model tests in eviction suite):**
- End-to-end inference with actual LLM
- GPU-side L2 norm computation in `vllm/v1/worker/gpu_model_runner.py` (verified by source inspection only)
- `vllm/v1/attention/backends/` flash_attn integration

## Test Types

**Unit Tests (pure logic):** `test_strategies.py`, `test_block_utils.py`, `test_segmenter.py` — no external dependencies, no I/O, deterministic inputs.

**Integration Tests (mocked dependencies):** `test_orchestrator.py` — tests the full orchestrator pipeline with mocked engine client and tokenizer.

**Structural/Wiring Tests (source inspection):** `test_serving_integration.py`, `test_l2_norm_delivery.py`, `test_no_eviction_guard.py` — verify integration points without instantiating full server or GPU workers.

**Protocol Validation Tests:** `test_protocol_extension.py` — pydantic model field defaults and validation rules.

**Behavioral Tests (monkey-patching):** `test_scheduler_eviction_fix.py` — uses `object.__new__(Scheduler)` to test a scheduler method without instantiating the full scheduler (bypasses CUDA dependencies).

**E2E Tests:** Not present in `tests/eviction/`. GPU-dependent e2e tests are in the CI Buildkite pipeline (`tests/v1/`, `tests/entrypoints/`) and are not run by the eviction test suite.

---

*Testing analysis: 2026-04-07*
