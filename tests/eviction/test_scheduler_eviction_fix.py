"""Tests confirming scheduler clears eviction data after processing.

Covers:
- SCHED-01: _process_evictions clears processed entries so ranges are not
            re-freed on subsequent scheduler ticks
"""
from __future__ import annotations

from unittest.mock import MagicMock, call, patch
import pytest


# ---------------------------------------------------------------------------
# Minimal scheduler-like fixture
# ---------------------------------------------------------------------------

def make_scheduler_stub(block_size: int = 16):
    """Return a minimal object mimicking the scheduler eviction interface.

    We monkey-patch only the attributes accessed by _process_evictions so
    the test does not need to import the full vLLM scheduler (which has heavy
    dependencies on CUDA / vLLM internals).
    """
    from vllm.v1.core.sched.scheduler import Scheduler

    mock_kv = MagicMock()
    mock_kv.block_size = block_size

    sched = object.__new__(Scheduler)
    sched.request_eviction_data = {}
    sched.kv_cache_manager = mock_kv
    return sched


# ---------------------------------------------------------------------------
# Test 1: After _process_evictions(), request_eviction_data is empty
# ---------------------------------------------------------------------------

def test_eviction_data_cleared_after_processing() -> None:
    """After _process_evictions runs, processed entries are removed."""
    sched = make_scheduler_stub(block_size=16)
    sched.request_eviction_data["req-1"] = [(0, 64)]

    sched._process_evictions()

    assert "req-1" not in sched.request_eviction_data


# ---------------------------------------------------------------------------
# Test 2: Second call is a no-op — free_blocks called only once
# ---------------------------------------------------------------------------

def test_double_call_does_not_re_free_blocks() -> None:
    """Calling _process_evictions twice does not re-free blocks."""
    sched = make_scheduler_stub(block_size=16)
    sched.request_eviction_data["req-2"] = [(0, 64)]

    sched._process_evictions()  # first call: frees blocks, clears data
    sched._process_evictions()  # second call: no data, should be a no-op

    # free_blocks should have been called exactly once
    assert sched.kv_cache_manager.free_blocks.call_count == 1


# ---------------------------------------------------------------------------
# Test 3: New data added after clearing is processed on the next tick
# ---------------------------------------------------------------------------

def test_new_data_after_clearing_is_processed() -> None:
    """After clearing, freshly added eviction data is processed next tick."""
    sched = make_scheduler_stub(block_size=16)
    sched.request_eviction_data["req-3"] = [(0, 64)]

    sched._process_evictions()  # clears req-3

    # Add new data after clearing
    sched.request_eviction_data["req-3"] = [(64, 128)]
    sched._process_evictions()  # should process new ranges

    assert "req-3" not in sched.request_eviction_data
    # free_blocks must have been called twice total (once per tick)
    assert sched.kv_cache_manager.free_blocks.call_count == 2


# ---------------------------------------------------------------------------
# Test 4: Unrelated request_ids are not prematurely cleared
# ---------------------------------------------------------------------------

def test_only_processed_entries_are_cleared() -> None:
    """_process_evictions only clears entries that were processed this tick."""
    sched = make_scheduler_stub(block_size=16)
    sched.request_eviction_data["req-4a"] = [(0, 32)]
    sched.request_eviction_data["req-4b"] = [(32, 64)]

    sched._process_evictions()

    # Both should be cleared (both were processed)
    assert "req-4a" not in sched.request_eviction_data
    assert "req-4b" not in sched.request_eviction_data
    # free_blocks called once per request with blocks
    assert sched.kv_cache_manager.free_blocks.call_count == 2
