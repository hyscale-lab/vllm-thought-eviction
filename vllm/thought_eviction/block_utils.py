"""
Block-aligned range utilities for KV cache eviction.

Provides three pure functions used by eviction strategies to prepare token
ranges for physical KV cache block eviction:

1. merge_overlapping_ranges — consolidate overlapping/adjacent ranges
2. align_ranges_to_blocks — snap ranges to block boundaries (ceil start, floor end)
3. apply_retention_window — trim ranges that extend into the retention zone

The correct ordering for eviction is: merge -> align -> retain.
Merging before alignment ensures that adjacent ranges that share a block do
not lose that block due to fragmentation (see Pitfall #3 in the design doc).
"""

from vllm.logger import init_logger

logger = init_logger(__name__)


def merge_overlapping_ranges(
    ranges: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Merge overlapping or adjacent token ranges.

    Args:
        ranges: List of (start, end) token index pairs (end is exclusive).

    Returns:
        New sorted list with all overlapping and adjacent ranges merged.
    """
    if not ranges:
        return []

    sorted_ranges = sorted(set(ranges))
    merged = [sorted_ranges[0]]

    for current_start, current_end in sorted_ranges[1:]:
        last_start, last_end = merged[-1]
        if current_start <= last_end:
            merged[-1] = (last_start, max(last_end, current_end))
        else:
            merged.append((current_start, current_end))

    return merged


def align_ranges_to_blocks(
    ranges: list[tuple[int, int]],
    block_size: int,
) -> list[tuple[int, int]]:
    """Align token ranges to KV cache block boundaries.

    Start indices are ceil-aligned (rounded up) and end indices are
    floor-aligned (rounded down) so that only fully-covered blocks are
    included. Ranges that do not cover at least one full block are dropped.
    The result is merged to avoid redundant adjacent spans.

    Args:
        ranges: List of (start, end) token index pairs.
        block_size: Number of tokens per KV cache block.

    Returns:
        Block-aligned, merged list of (start, end) pairs.
    """
    aligned: list[tuple[int, int]] = []
    for start, end in ranges:
        aligned_start = (start + block_size - 1) // block_size * block_size
        aligned_end = end // block_size * block_size

        if aligned_end > aligned_start:
            aligned.append((aligned_start, aligned_end))

    return merge_overlapping_ranges(aligned)


def apply_retention_window(
    ranges: list[tuple[int, int]],
    retention_floor: int,
) -> list[tuple[int, int]]:
    """Trim ranges to stay below the retention floor.

    Ranges fully below the floor are kept unchanged. Ranges that cross the
    floor are truncated at the floor. Ranges fully above the floor are
    discarded.

    Args:
        ranges: List of (start, end) token index pairs.
        retention_floor: Token index below which eviction is permitted.

    Returns:
        Trimmed list of ranges within the retention window.
    """
    trimmed: list[tuple[int, int]] = []
    for start, end in ranges:
        if end <= retention_floor:
            trimmed.append((start, end))
        elif start < retention_floor:
            trimmed.append((start, retention_floor))
        # else: fully above floor — discard
    return trimmed
