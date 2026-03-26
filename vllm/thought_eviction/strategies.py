"""
Eviction strategy classes for server-side KV cache management.

This module provides four strategy classes that determine which tokens or
thoughts to evict based on L2 norms or random scoring. All strategies return
reasoning-relative token ranges (start_token_pos values starting at 0) — the
orchestrator in Phase 2 applies the absolute sequence offset.

Strategies:
    GlobalStrategy:     Evict tokens with highest L2 norms globally across all
                        reasoning tokens.
    ThoughtMinStrategy: Evict thoughts with highest minimum L2 norm per thought.
    ThoughtAvgStrategy: Evict thoughts with highest average L2 norm per thought.
    RandomStrategy:     Evict randomly-scored thoughts with stable per-request
                        scores keyed by start_char_pos.
"""

import random
from typing import Optional

import numpy as np

from vllm.logger import init_logger
from vllm.thought_eviction.segmenter import ThoughtSegment

logger = init_logger(__name__)


def _indices_to_ranges(indices: list[int]) -> list[tuple[int, int]]:
    """Convert a list of token indices to consecutive (start, end) ranges.

    Consecutive indices are merged into a single range. All ranges are
    reasoning-relative (no offset added).

    Args:
        indices: Unsorted list of token indices to convert.

    Returns:
        Sorted list of (start, end) tuples where end is exclusive.
    """
    if not indices:
        return []

    indices = sorted(indices)
    ranges = []
    start = indices[0]
    end = indices[0] + 1

    for idx in indices[1:]:
        if idx == end:
            end = idx + 1
        else:
            ranges.append((start, end))
            start = idx
            end = idx + 1

    ranges.append((start, end))
    return ranges


class GlobalStrategy:
    """Evict tokens with highest L2 norms globally.

    Sorts all reasoning tokens by L2 norm and evicts the top
    (1 - keep_ratio) fraction. Token positions are reasoning-relative.
    """

    def compute_evictable_ranges(
        self,
        l2_norms: np.ndarray,
        keep_ratio: float,
        prune_after_tokens: int,
    ) -> list[tuple[int, int]]:
        """Compute evictable token ranges using global L2 norm ranking.

        Args:
            l2_norms: Array of L2 norms for all reasoning tokens.
            keep_ratio: Fraction of tokens to keep (0.0 to 1.0).
            prune_after_tokens: Minimum token count before eviction begins.

        Returns:
            List of reasoning-relative (start, end) ranges to evict.
            Empty list if not enough tokens or keep_ratio is 1.0.
        """
        if len(l2_norms) < prune_after_tokens:
            return []

        num_tokens = len(l2_norms)
        tokens_to_keep = int(keep_ratio * num_tokens)

        # Get indices sorted by L2 norm ascending — keep lowest norms
        sorted_indices = np.argsort(l2_norms)

        # Indices to evict are those NOT in the lowest tokens_to_keep norms
        indices_to_keep = set(sorted_indices[:tokens_to_keep].tolist())
        indices_to_evict = [i for i in range(num_tokens) if i not in indices_to_keep]

        return _indices_to_ranges(indices_to_evict)


class ThoughtMinStrategy:
    """Evict thoughts with highest minimum L2 norm.

    Thoughts with low minimum norms likely contain attention sinks and
    are more important to retain. Thoughts are sorted by their minimum
    norm value and the top (1 - keep_ratio) fraction are evicted.
    """

    def compute_evictable_ranges(
        self,
        thoughts: list[ThoughtSegment],
        keep_ratio: float,
        min_segment_tokens: int,
        prune_after_tokens: int,
        protect_first_thought: bool = True,
    ) -> list[tuple[int, int]]:
        """Compute evictable token ranges by sorting thoughts on min L2 norm.

        Args:
            thoughts: List of ThoughtSegment objects with l2_norms populated.
            keep_ratio: Fraction of eviction candidates to keep (0.0 to 1.0).
            min_segment_tokens: Minimum tokens required for a thought to be
                considered a valid eviction candidate.
            prune_after_tokens: Minimum total token count across valid thoughts
                before eviction begins.
            protect_first_thought: If True, the first valid thought is never
                added to the eviction candidate pool.

        Returns:
            List of reasoning-relative (start, end) ranges to evict.
        """
        valid_thoughts = [
            t for t in thoughts
            if (t.end_token_pos - t.start_token_pos) >= min_segment_tokens
            and t.l2_norms is not None
            and len(t.l2_norms) > 0
        ]

        if len(valid_thoughts) < 2:
            return []

        eviction_candidates = valid_thoughts[1:] if protect_first_thought else valid_thoughts

        if len(eviction_candidates) < 1:
            return []

        total_tokens = sum(t.end_token_pos - t.start_token_pos for t in valid_thoughts)
        if total_tokens < prune_after_tokens:
            return []

        # Sort by minimum L2 norm ascending — keep those with lowest min norms
        sorted_thoughts = sorted(eviction_candidates, key=lambda t: t.min_l2_norm)

        num_to_keep = max(1, int(keep_ratio * len(sorted_thoughts)))
        thoughts_to_evict = sorted_thoughts[num_to_keep:]

        evictable_ranges = []
        for thought in thoughts_to_evict:
            thought.evicted = True
            evictable_ranges.append((thought.start_token_pos, thought.end_token_pos))

        return evictable_ranges


class ThoughtAvgStrategy:
    """Evict thoughts with highest average L2 norm.

    Thoughts with low average norms are likely more important overall.
    Thoughts are sorted by their average norm value and the top
    (1 - keep_ratio) fraction are evicted.
    """

    def compute_evictable_ranges(
        self,
        thoughts: list[ThoughtSegment],
        keep_ratio: float,
        min_segment_tokens: int,
        prune_after_tokens: int,
        protect_first_thought: bool = True,
    ) -> list[tuple[int, int]]:
        """Compute evictable token ranges by sorting thoughts on avg L2 norm.

        Args:
            thoughts: List of ThoughtSegment objects with l2_norms populated.
            keep_ratio: Fraction of eviction candidates to keep (0.0 to 1.0).
            min_segment_tokens: Minimum tokens required for a thought to be
                considered a valid eviction candidate.
            prune_after_tokens: Minimum total token count across valid thoughts
                before eviction begins.
            protect_first_thought: If True, the first valid thought is never
                added to the eviction candidate pool.

        Returns:
            List of reasoning-relative (start, end) ranges to evict.
        """
        valid_thoughts = [
            t for t in thoughts
            if (t.end_token_pos - t.start_token_pos) >= min_segment_tokens
            and t.l2_norms is not None
            and len(t.l2_norms) > 0
        ]

        if len(valid_thoughts) < 2:
            return []

        eviction_candidates = valid_thoughts[1:] if protect_first_thought else valid_thoughts

        if len(eviction_candidates) < 1:
            return []

        total_tokens = sum(t.end_token_pos - t.start_token_pos for t in valid_thoughts)
        if total_tokens < prune_after_tokens:
            return []

        # Sort by average L2 norm ascending — keep those with lowest avg norms
        sorted_thoughts = sorted(eviction_candidates, key=lambda t: t.avg_l2_norm)

        num_to_keep = max(1, int(keep_ratio * len(sorted_thoughts)))
        thoughts_to_evict = sorted_thoughts[num_to_keep:]

        evictable_ranges = []
        for thought in thoughts_to_evict:
            thought.evicted = True
            evictable_ranges.append((thought.start_token_pos, thought.end_token_pos))

        return evictable_ranges


class RandomStrategy:
    """Evict randomly-scored thoughts with stable per-request scores.

    Each thought is assigned a random score once on first encounter,
    keyed by start_char_pos. Subsequent eviction cycles reuse the same
    scores so the same thoughts are always ranked in the evict tier —
    matching the stable, cycle-invariant selection of ThoughtMinStrategy
    without L2 norm computation overhead.

    Call reset_scores() between requests to clear the score cache.
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        """Initialize with an optional RNG seed for reproducibility.

        Args:
            seed: Optional integer seed for random.Random. If None, a
                non-deterministic seed is used.
        """
        self.rng = random.Random(seed)
        self._thought_random_scores: dict[int, float] = {}

    def compute_evictable_ranges(
        self,
        thoughts: list[ThoughtSegment],
        keep_ratio: float,
        min_segment_tokens: int,
        prune_after_tokens: int,
        protect_first_thought: bool = True,
    ) -> list[tuple[int, int]]:
        """Compute evictable token ranges using stable random thought scores.

        Scores are assigned once per thought (keyed by start_char_pos) and
        reused across calls, ensuring stable eviction selection per request.

        Args:
            thoughts: List of ThoughtSegment objects.
            keep_ratio: Fraction of eviction candidates to keep (0.0 to 1.0).
            min_segment_tokens: Minimum tokens required for a thought to be
                considered a valid eviction candidate.
            prune_after_tokens: Minimum total token count across valid thoughts
                before eviction begins.
            protect_first_thought: If True, the first valid thought is never
                added to the eviction candidate pool.

        Returns:
            List of reasoning-relative (start, end) ranges to evict.
        """
        valid_thoughts = [
            t for t in thoughts
            if (t.end_token_pos - t.start_token_pos) >= min_segment_tokens
        ]

        if len(valid_thoughts) < 2:
            return []

        eviction_candidates = valid_thoughts[1:] if protect_first_thought else valid_thoughts

        if len(eviction_candidates) < 1:
            return []

        total_tokens = sum(t.end_token_pos - t.start_token_pos for t in valid_thoughts)
        if total_tokens < prune_after_tokens:
            return []

        # Assign a stable random score to each thought on first encounter
        for thought in eviction_candidates:
            key = thought.start_char_pos
            if key not in self._thought_random_scores:
                self._thought_random_scores[key] = self.rng.random()

        # Sort by stable score — same ranking every cycle
        sorted_candidates = sorted(
            eviction_candidates,
            key=lambda t: self._thought_random_scores[t.start_char_pos],
        )

        num_to_keep = max(1, int(keep_ratio * len(sorted_candidates)))
        thoughts_to_evict = sorted_candidates[num_to_keep:]

        if not thoughts_to_evict:
            return []

        evictable_ranges = []
        for thought in thoughts_to_evict:
            thought.evicted = True
            evictable_ranges.append((thought.start_token_pos, thought.end_token_pos))

        return evictable_ranges

    def reset_scores(self) -> None:
        """Clear stable scores for a new request."""
        self._thought_random_scores.clear()
