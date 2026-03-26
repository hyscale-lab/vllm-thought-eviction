"""
Unit tests for vllm.thought_eviction.strategies.

Coverage:
- STRAT-01: GlobalStrategy sorts by L2 norm and evicts highest-norm tokens
- STRAT-02: ThoughtMinStrategy sorts thoughts by min norm and evicts fraction
- STRAT-03: ThoughtAvgStrategy sorts thoughts by avg norm and evicts fraction
- STRAT-04: RandomStrategy produces stable scores keyed on start_char_pos
- Cross-cutting: all strategies return reasoning-relative ranges
"""

import numpy as np
import pytest

from vllm.thought_eviction.segmenter import ThoughtSegment
from vllm.thought_eviction.strategies import (
    GlobalStrategy,
    ThoughtAvgStrategy,
    ThoughtMinStrategy,
    RandomStrategy,
    _indices_to_ranges,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_thought(
    text: str,
    start_char: int,
    end_char: int,
    start_tok: int,
    end_tok: int,
    min_norm: float = float("inf"),
    avg_norm: float = float("inf"),
    l2_norms: np.ndarray | None = None,
) -> ThoughtSegment:
    """Build a ThoughtSegment with known field values for testing."""
    if l2_norms is None:
        l2_norms = np.array([min_norm])
    return ThoughtSegment(
        text=text,
        start_char_pos=start_char,
        end_char_pos=end_char,
        start_token_pos=start_tok,
        end_token_pos=end_tok,
        l2_norms=l2_norms,
        min_l2_norm=min_norm,
        avg_l2_norm=avg_norm,
    )


def make_thought_no_l2(
    text: str,
    start_char: int,
    end_char: int,
    start_tok: int,
    end_tok: int,
) -> ThoughtSegment:
    """Build a ThoughtSegment without L2 norms (for testing filtering)."""
    return ThoughtSegment(
        text=text,
        start_char_pos=start_char,
        end_char_pos=end_char,
        start_token_pos=start_tok,
        end_token_pos=end_tok,
        l2_norms=None,
    )


# ---------------------------------------------------------------------------
# _indices_to_ranges helper
# ---------------------------------------------------------------------------

def test_indices_to_ranges_empty():
    assert _indices_to_ranges([]) == []


def test_indices_to_ranges_single():
    assert _indices_to_ranges([3]) == [(3, 4)]


def test_indices_to_ranges_consecutive():
    assert _indices_to_ranges([1, 2, 3]) == [(1, 4)]


def test_indices_to_ranges_disjoint():
    result = _indices_to_ranges([0, 2, 4])
    assert result == [(0, 1), (2, 3), (4, 5)]


def test_indices_to_ranges_mixed():
    result = _indices_to_ranges([0, 4, 6, 8, 2])
    # After sort: [0, 2, 4, 6, 8]
    assert result == [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9)]


# ---------------------------------------------------------------------------
# GlobalStrategy — STRAT-01
# ---------------------------------------------------------------------------

def test_global_strategy_evicts_highest_norms():
    """Test 1: 10 tokens, keep_ratio=0.5 — lowest 5 norms kept, rest evicted."""
    strategy = GlobalStrategy()
    # norms: 0.5, 0.1, 0.9, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6, 0.05
    # sorted ascending: idx 9(0.05), 1(0.1), 5(0.2), 3(0.3), 7(0.4)  <- kept
    # evicted: idx 0(0.5), 8(0.6), 4(0.7), 6(0.8), 2(0.9)
    norms = np.array([0.5, 0.1, 0.9, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6, 0.05])
    ranges = strategy.compute_evictable_ranges(
        l2_norms=norms,
        keep_ratio=0.5,
        prune_after_tokens=1,
    )

    # Collect all evicted indices from ranges
    evicted_indices = set()
    for start, end in ranges:
        evicted_indices.update(range(start, end))

    kept_indices = set(range(10)) - evicted_indices

    # Verify 5 tokens kept with lowest norms
    assert kept_indices == {9, 1, 5, 3, 7}, f"Expected lowest 5 norms kept, got {kept_indices}"
    assert evicted_indices == {0, 8, 4, 6, 2}, f"Expected highest 5 norms evicted, got {evicted_indices}"


def test_global_strategy_below_prune_threshold_returns_empty():
    """Test 2: len(l2_norms) < prune_after_tokens returns []."""
    strategy = GlobalStrategy()
    norms = np.array([0.1, 0.2, 0.3])
    ranges = strategy.compute_evictable_ranges(
        l2_norms=norms,
        keep_ratio=0.5,
        prune_after_tokens=10,
    )
    assert ranges == []


def test_global_strategy_keep_all_returns_empty():
    """Test 3: keep_ratio=1.0 returns [] (keep everything)."""
    strategy = GlobalStrategy()
    norms = np.array([0.5, 0.1, 0.9, 0.3, 0.7])
    ranges = strategy.compute_evictable_ranges(
        l2_norms=norms,
        keep_ratio=1.0,
        prune_after_tokens=1,
    )
    assert ranges == []


def test_global_strategy_ranges_are_reasoning_relative():
    """Test 12 (cross-cutting): Global ranges start at 0, not offset-adjusted."""
    strategy = GlobalStrategy()
    norms = np.array([0.9, 0.1, 0.8, 0.2])  # keep_ratio=0.5 → keep idx 1,3; evict idx 0,2
    ranges = strategy.compute_evictable_ranges(
        l2_norms=norms,
        keep_ratio=0.5,
        prune_after_tokens=1,
    )
    # All range values should be within [0, len(norms)]
    for start, end in ranges:
        assert 0 <= start < len(norms), f"Range start {start} out of bounds"
        assert 0 < end <= len(norms), f"Range end {end} out of bounds"


# ---------------------------------------------------------------------------
# ThoughtMinStrategy — STRAT-02
# ---------------------------------------------------------------------------

def test_thought_min_strategy_sorts_by_min_norm():
    """Test 4: 4 thoughts, protect_first=True, keep_ratio=0.5.

    Setup:
        thought0: min=0.1 (protected, first)
        thought1: min=0.5
        thought2: min=0.2
        thought3: min=0.8

    Candidates: [thought1(0.5), thought2(0.2), thought3(0.8)]
    After sort ascending: [thought2(0.2), thought1(0.5), thought3(0.8)]
    keep=max(1, int(0.5*3))=1 → keep thought2, evict thought1 and thought3
    """
    strategy = ThoughtMinStrategy()
    t0 = make_thought("t0", 0, 50, 0, 20, min_norm=0.1, avg_norm=0.1, l2_norms=np.full(20, 0.1))
    t1 = make_thought("t1", 50, 100, 20, 40, min_norm=0.5, avg_norm=0.5, l2_norms=np.full(20, 0.5))
    t2 = make_thought("t2", 100, 150, 40, 60, min_norm=0.2, avg_norm=0.2, l2_norms=np.full(20, 0.2))
    t3 = make_thought("t3", 150, 200, 60, 80, min_norm=0.8, avg_norm=0.8, l2_norms=np.full(20, 0.8))

    ranges = strategy.compute_evictable_ranges(
        thoughts=[t0, t1, t2, t3],
        keep_ratio=0.5,
        min_segment_tokens=5,
        prune_after_tokens=1,
        protect_first_thought=True,
    )

    # thought1 (token 20-40) and thought3 (token 60-80) should be evicted
    evicted_starts = {r[0] for r in ranges}
    assert 20 in evicted_starts, "thought1 (start=20) should be evicted"
    assert 60 in evicted_starts, "thought3 (start=60) should be evicted"
    assert 40 not in evicted_starts, "thought2 (start=40) should be kept"
    assert 0 not in evicted_starts, "thought0 (start=0) should be protected"


def test_thought_min_strategy_too_few_valid_thoughts():
    """Test 5: fewer than 2 valid thoughts returns []."""
    strategy = ThoughtMinStrategy()
    t0 = make_thought("t0", 0, 50, 0, 20, min_norm=0.1, l2_norms=np.full(20, 0.1))
    # t1 too small (only 3 tokens, below min_segment_tokens=5)
    t1 = make_thought("t1", 50, 70, 20, 23, min_norm=0.5, l2_norms=np.full(3, 0.5))

    ranges = strategy.compute_evictable_ranges(
        thoughts=[t0, t1],
        keep_ratio=0.5,
        min_segment_tokens=5,
        prune_after_tokens=1,
    )
    assert ranges == []


def test_thought_min_strategy_below_prune_threshold():
    """Test 6: total_tokens < prune_after_tokens returns []."""
    strategy = ThoughtMinStrategy()
    t0 = make_thought("t0", 0, 50, 0, 10, min_norm=0.1, l2_norms=np.full(10, 0.1))
    t1 = make_thought("t1", 50, 100, 10, 20, min_norm=0.5, l2_norms=np.full(10, 0.5))

    ranges = strategy.compute_evictable_ranges(
        thoughts=[t0, t1],
        keep_ratio=0.5,
        min_segment_tokens=5,
        prune_after_tokens=1000,  # far above total tokens
    )
    assert ranges == []


def test_thought_min_strategy_ranges_are_reasoning_relative():
    """Ranges use start_token_pos directly, not offset-adjusted."""
    strategy = ThoughtMinStrategy()
    t0 = make_thought("t0", 0, 50, 5, 25, min_norm=0.1, l2_norms=np.full(20, 0.1))
    t1 = make_thought("t1", 50, 100, 25, 45, min_norm=0.9, l2_norms=np.full(20, 0.9))

    ranges = strategy.compute_evictable_ranges(
        thoughts=[t0, t1],
        keep_ratio=0.5,
        min_segment_tokens=5,
        prune_after_tokens=1,
        protect_first_thought=True,
    )

    # With protect_first=True: only t1 is a candidate; keep=max(1, int(0.5*1))=1 → keep t1
    # So nothing evicted with 1 candidate and keep_ratio=0.5
    # Use protect_first=False to actually evict
    ranges2 = strategy.compute_evictable_ranges(
        thoughts=[t0, t1],
        keep_ratio=0.0,
        min_segment_tokens=5,
        prune_after_tokens=1,
        protect_first_thought=False,
    )

    for start, end in ranges2:
        # start_token_pos values are 5 and 25, not 5+N
        assert start in {5, 25}, f"Range start {start} should match start_token_pos (5 or 25)"


# ---------------------------------------------------------------------------
# ThoughtAvgStrategy — STRAT-03
# ---------------------------------------------------------------------------

def test_thought_avg_strategy_sorts_by_avg_norm():
    """Test 7: Same structure as Test 4 but sorting by avg_l2_norm.

    Setup:
        thought0: avg=0.1 (protected, first)
        thought1: avg=0.5
        thought2: avg=0.2
        thought3: avg=0.8

    Candidates: [t1(0.5), t2(0.2), t3(0.8)]
    After sort ascending: [t2(0.2), t1(0.5), t3(0.8)]
    keep=max(1, int(0.5*3))=1 → keep t2, evict t1 and t3
    """
    strategy = ThoughtAvgStrategy()
    t0 = make_thought("t0", 0, 50, 0, 20, min_norm=0.1, avg_norm=0.1, l2_norms=np.full(20, 0.1))
    t1 = make_thought("t1", 50, 100, 20, 40, min_norm=0.5, avg_norm=0.5, l2_norms=np.full(20, 0.5))
    t2 = make_thought("t2", 100, 150, 40, 60, min_norm=0.2, avg_norm=0.2, l2_norms=np.full(20, 0.2))
    t3 = make_thought("t3", 150, 200, 60, 80, min_norm=0.8, avg_norm=0.8, l2_norms=np.full(20, 0.8))

    ranges = strategy.compute_evictable_ranges(
        thoughts=[t0, t1, t2, t3],
        keep_ratio=0.5,
        min_segment_tokens=5,
        prune_after_tokens=1,
        protect_first_thought=True,
    )

    evicted_starts = {r[0] for r in ranges}
    assert 20 in evicted_starts, "thought1 (start=20) should be evicted"
    assert 60 in evicted_starts, "thought3 (start=60) should be evicted"
    assert 40 not in evicted_starts, "thought2 (start=40) should be kept"
    assert 0 not in evicted_starts, "thought0 (start=0) should be protected"


def test_thought_avg_strategy_uses_avg_not_min():
    """Verify avg strategy sorts by avg_l2_norm, not min_l2_norm."""
    strategy = ThoughtAvgStrategy()
    # t1: min=0.1, avg=0.9 (should be evicted by avg but kept by min)
    # t2: min=0.9, avg=0.1 (should be kept by avg but evicted by min)
    t0 = make_thought("t0", 0, 50, 0, 20, min_norm=0.5, avg_norm=0.5, l2_norms=np.full(20, 0.5))
    t1 = make_thought("t1", 50, 100, 20, 40, min_norm=0.1, avg_norm=0.9, l2_norms=np.array([0.1] + [0.95] * 19))
    t2 = make_thought("t2", 100, 150, 40, 60, min_norm=0.9, avg_norm=0.1, l2_norms=np.array([0.9] + [0.05] * 19))

    # 2 candidates, keep_ratio=0.5 → keep 1, evict 1
    # sorted by avg: [t2(avg=0.1), t1(avg=0.9)] → keep t2, evict t1
    ranges = strategy.compute_evictable_ranges(
        thoughts=[t0, t1, t2],
        keep_ratio=0.5,
        min_segment_tokens=5,
        prune_after_tokens=1,
        protect_first_thought=True,
    )

    evicted_starts = {r[0] for r in ranges}
    assert 20 in evicted_starts, "t1 (high avg=0.9) should be evicted"
    assert 40 not in evicted_starts, "t2 (low avg=0.1) should be kept"


# ---------------------------------------------------------------------------
# RandomStrategy — STRAT-04
# ---------------------------------------------------------------------------

def test_random_strategy_stable_scores():
    """Test 8: Calling compute_evictable_ranges twice gives identical results."""
    strategy = RandomStrategy(seed=42)
    thoughts = [
        make_thought("t0", 0, 50, 0, 20),
        make_thought("t1", 50, 100, 20, 40),
        make_thought("t2", 100, 150, 40, 60),
        make_thought("t3", 150, 200, 60, 80),
    ]

    # Reset evicted flags before second call
    for t in thoughts:
        t.evicted = False

    ranges1 = strategy.compute_evictable_ranges(
        thoughts=thoughts,
        keep_ratio=0.5,
        min_segment_tokens=5,
        prune_after_tokens=1,
        protect_first_thought=True,
    )

    # Reset evicted flags for second call
    for t in thoughts:
        t.evicted = False

    ranges2 = strategy.compute_evictable_ranges(
        thoughts=thoughts,
        keep_ratio=0.5,
        min_segment_tokens=5,
        prune_after_tokens=1,
        protect_first_thought=True,
    )

    assert ranges1 == ranges2, f"Expected identical ranges, got {ranges1} vs {ranges2}"


def test_random_scores_keyed_on_start_char_pos():
    """Test 9: Scores in _thought_random_scores dict are keyed by start_char_pos."""
    strategy = RandomStrategy(seed=7)
    thoughts = [
        make_thought("t0", 0, 50, 0, 20),
        make_thought("t1", 100, 150, 20, 40),  # start_char_pos=100
        make_thought("t2", 200, 250, 40, 60),  # start_char_pos=200
    ]

    strategy.compute_evictable_ranges(
        thoughts=thoughts,
        keep_ratio=0.5,
        min_segment_tokens=5,
        prune_after_tokens=1,
        protect_first_thought=True,
    )

    # Scores should be keyed on start_char_pos (100, 200), not indices
    assert 100 in strategy._thought_random_scores, "Score for start_char_pos=100 should exist"
    assert 200 in strategy._thought_random_scores, "Score for start_char_pos=200 should exist"
    assert 0 not in strategy._thought_random_scores, "Protected thought (t0) should not have score"


def test_random_strategy_different_seed_different_selection():
    """Test 10: Different seed may produce different selection."""
    thoughts = [
        make_thought("t0", 0, 50, 0, 20),
        make_thought("t1", 50, 100, 20, 40),
        make_thought("t2", 100, 150, 40, 60),
        make_thought("t3", 150, 200, 60, 80),
        make_thought("t4", 200, 250, 80, 100),
        make_thought("t5", 250, 300, 100, 120),
    ]

    results = set()
    for seed in range(20):
        s = RandomStrategy(seed=seed)
        for t in thoughts:
            t.evicted = False
        ranges = s.compute_evictable_ranges(
            thoughts=thoughts,
            keep_ratio=0.5,
            min_segment_tokens=5,
            prune_after_tokens=1,
        )
        results.add(tuple(sorted(ranges)))

    # With 20 different seeds, at least 2 distinct results expected
    assert len(results) > 1, "Different seeds should produce different selections"


def test_random_strategy_protect_first_thought():
    """Test 11: protect_first_thought=True skips first valid thought."""
    strategy = RandomStrategy(seed=42)
    thoughts = [
        make_thought("t0", 0, 50, 0, 20),    # first valid — should be protected
        make_thought("t1", 50, 100, 20, 40),
        make_thought("t2", 100, 150, 40, 60),
    ]

    for _ in range(5):
        for t in thoughts:
            t.evicted = False
        ranges = strategy.compute_evictable_ranges(
            thoughts=thoughts,
            keep_ratio=0.5,
            min_segment_tokens=5,
            prune_after_tokens=1,
            protect_first_thought=True,
        )
        evicted_starts = {r[0] for r in ranges}
        assert 0 not in evicted_starts, "First thought (start=0) should never be evicted"


def test_random_strategy_reset_scores():
    """reset_scores() clears the score cache."""
    strategy = RandomStrategy(seed=42)
    thoughts = [
        make_thought("t0", 0, 50, 0, 20),
        make_thought("t1", 50, 100, 20, 40),
        make_thought("t2", 100, 150, 40, 60),
    ]

    strategy.compute_evictable_ranges(
        thoughts=thoughts,
        keep_ratio=0.5,
        min_segment_tokens=5,
        prune_after_tokens=1,
        protect_first_thought=True,
    )
    assert len(strategy._thought_random_scores) > 0

    strategy.reset_scores()
    assert strategy._thought_random_scores == {}


# ---------------------------------------------------------------------------
# Cross-cutting: all strategies return reasoning-relative ranges
# ---------------------------------------------------------------------------

def test_thought_min_strategy_ranges_match_token_positions():
    """All thought strategies return start_token_pos values, not offset-adjusted."""
    strategy = ThoughtMinStrategy()
    # Thoughts with non-zero token positions to verify no offset is added
    t0 = make_thought("t0", 0, 50, 100, 120, min_norm=0.1, l2_norms=np.full(20, 0.1))
    t1 = make_thought("t1", 50, 100, 120, 140, min_norm=0.9, l2_norms=np.full(20, 0.9))

    ranges = strategy.compute_evictable_ranges(
        thoughts=[t0, t1],
        keep_ratio=0.0,  # evict everything
        min_segment_tokens=5,
        prune_after_tokens=1,
        protect_first_thought=False,
    )

    # Sort by min, evict all: t0(0.1) and t1(0.9) — but keep=max(1, 0)=1
    # So only t1 evicted (highest norm)
    assert len(ranges) >= 1
    for start, end in ranges:
        # Must equal the actual start/end_token_pos, no offset added
        assert start in {100, 120}, f"start {start} should be a token position (100 or 120)"
        assert end in {120, 140}, f"end {end} should be a token position (120 or 140)"
