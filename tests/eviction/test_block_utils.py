"""
Unit tests for vllm.thought_eviction.block_utils.

Coverage:
- merge_overlapping_ranges: overlapping, adjacent, empty input
- align_ranges_to_blocks: basic alignment, ranges smaller than a block
- merge-before-align ordering: critical Pitfall #3
- apply_retention_window: trimming, fully above floor
"""

import pytest

from vllm.thought_eviction.block_utils import (
    merge_overlapping_ranges,
    align_ranges_to_blocks,
    apply_retention_window,
)


# ---------------------------------------------------------------------------
# merge_overlapping_ranges
# ---------------------------------------------------------------------------

class TestMergeOverlappingRanges:
    """merge_overlapping_ranges consolidates overlapping and adjacent ranges."""

    def test_overlapping_ranges_merged(self):
        """Overlapping ranges are merged into a single span."""
        result = merge_overlapping_ranges([(0, 10), (5, 15)])
        assert result == [(0, 15)]

    def test_adjacent_ranges_merged(self):
        """Adjacent (touching) ranges are merged."""
        result = merge_overlapping_ranges([(0, 10), (10, 20)])
        assert result == [(0, 20)]

    def test_empty_input_returns_empty(self):
        """Empty input returns empty list."""
        result = merge_overlapping_ranges([])
        assert result == []

    def test_disjoint_ranges_kept_separate(self):
        """Non-overlapping, non-adjacent ranges are kept as-is."""
        result = merge_overlapping_ranges([(0, 10), (20, 30)])
        assert result == [(0, 10), (20, 30)]

    def test_duplicate_ranges_deduplicated(self):
        """Duplicate ranges are treated as a single range."""
        result = merge_overlapping_ranges([(5, 10), (5, 10)])
        assert result == [(5, 10)]

    def test_unsorted_input_sorted_correctly(self):
        """Unsorted ranges are sorted before merging."""
        result = merge_overlapping_ranges([(20, 30), (0, 10), (5, 25)])
        assert result == [(0, 30)]


# ---------------------------------------------------------------------------
# align_ranges_to_blocks
# ---------------------------------------------------------------------------

class TestAlignRangesToBlocks:
    """align_ranges_to_blocks snaps ranges to block boundaries."""

    def test_range_aligned_to_block_boundaries(self):
        """A range spanning more than one block is correctly aligned.

        (5, 35) with block_size=16:
          aligned_start = ceil(5/16)*16 = 16
          aligned_end   = floor(35/16)*16 = 32
        """
        result = align_ranges_to_blocks([(5, 35)], block_size=16)
        assert result == [(16, 32)]

    def test_range_smaller_than_block_dropped(self):
        """A range that does not cover a full block is dropped."""
        result = align_ranges_to_blocks([(0, 8)], block_size=16)
        assert result == []

    def test_exact_block_boundary_range(self):
        """A range already on exact block boundaries passes through."""
        result = align_ranges_to_blocks([(0, 32)], block_size=16)
        assert result == [(0, 32)]

    def test_empty_input_returns_empty(self):
        """Empty input returns empty list."""
        result = align_ranges_to_blocks([], block_size=16)
        assert result == []

    def test_multiple_ranges_aligned_and_merged(self):
        """Multiple aligned ranges are merged if they become adjacent."""
        # (0, 32) aligned -> (0, 32); (32, 64) aligned -> (32, 64); merged -> (0, 64)
        result = align_ranges_to_blocks([(0, 32), (32, 64)], block_size=16)
        assert result == [(0, 64)]


# ---------------------------------------------------------------------------
# merge-before-align pitfall (Pitfall #3)
# ---------------------------------------------------------------------------

class TestMergeBeforeAlignPreservesSharedBlocks:
    """Critical ordering: merge BEFORE align must preserve shared block coverage."""

    def test_merge_before_align_preserves_shared_blocks(self):
        """Adjacent ranges that share a block must be merged first to preserve it.

        (100, 150) and (150, 200) with block_size=16:
          If aligned separately:
            (100, 150) -> aligned_start=ceil(100/16)*16=112, aligned_end=floor(150/16)*16=144 -> (112, 144)
            (150, 200) -> aligned_start=ceil(150/16)*16=160, aligned_end=floor(200/16)*16=192 -> (160, 192)
            Gap at (144, 160): block 144-160 is LOST

          If merged first:
            merged -> (100, 200)
            aligned -> aligned_start=112, aligned_end=192 -> (112, 192)
            Block 144-160 is PRESERVED
        """
        # Correct: merge first, then align
        merged_first = merge_overlapping_ranges([(100, 150), (150, 200)])
        result = align_ranges_to_blocks(merged_first, block_size=16)

        # The shared block 144-160 must be covered
        covered_tokens = set()
        for start, end in result:
            covered_tokens.update(range(start, end))

        assert 144 in covered_tokens, f"Token 144 should be covered; got ranges {result}"
        assert 159 in covered_tokens, f"Token 159 should be covered; got ranges {result}"

    def test_align_without_merge_loses_shared_block(self):
        """Without merging first, aligning separately loses the shared block."""
        # Aligned separately (wrong order)
        result = align_ranges_to_blocks([(100, 150), (150, 200)], block_size=16)

        # Block 144-160 is split across the two ranges and may be lost
        # This test documents the pitfall: the gap at 144-160 means token 152 is lost
        covered_tokens = set()
        for start, end in result:
            covered_tokens.update(range(start, end))

        # 152 is in the "gap" between the two aligned ranges (144-160)
        assert 152 not in covered_tokens, (
            "Without merge-first, token 152 in the gap block should be missing"
        )


# ---------------------------------------------------------------------------
# apply_retention_window
# ---------------------------------------------------------------------------

class TestApplyRetentionWindow:
    """apply_retention_window trims ranges to stay below the retention floor."""

    def test_apply_retention_window_trims_crossing_range(self):
        """A range that crosses the retention floor is trimmed at the floor."""
        result = apply_retention_window([(0, 100), (200, 300)], retention_floor=250)
        assert result == [(0, 100), (200, 250)]

    def test_apply_retention_window_drops_above_floor(self):
        """Ranges fully above the floor are dropped."""
        result = apply_retention_window([(300, 400)], retention_floor=250)
        assert result == []

    def test_apply_retention_window_keeps_below_floor(self):
        """Ranges fully below the floor are kept unchanged."""
        result = apply_retention_window([(0, 100), (100, 200)], retention_floor=300)
        assert result == [(0, 100), (100, 200)]

    def test_apply_retention_window_empty_input(self):
        """Empty input returns empty list."""
        result = apply_retention_window([], retention_floor=100)
        assert result == []

    def test_apply_retention_window_at_exact_floor(self):
        """Range ending exactly at the floor is kept (end <= floor)."""
        result = apply_retention_window([(0, 100)], retention_floor=100)
        assert result == [(0, 100)]
