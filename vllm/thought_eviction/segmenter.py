"""
Thought segmentation for server-side eviction.

This module provides ThoughtSegment and ThoughtSegmenter for detecting thought
boundaries within reasoning content and computing token-relative positions.

ThoughtSegmenter parses <think>...</think> tagged content and splits it into
discrete "thoughts" using 14 target linguistic boundary phrases. Token positions
are computed using offset_mapping from the tokenizer and are relative to the
start of the reasoning content (not absolute sequence positions).
"""

import regex as re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class ThoughtSegment:
    """Represents a segment of text identified as a 'thought'."""

    text: str
    start_char_pos: int
    end_char_pos: int
    start_token_pos: int   # Relative to start of reasoning_content
    end_token_pos: int     # Exclusive end
    l2_norms: Optional[np.ndarray] = None
    min_l2_norm: float = float('inf')
    avg_l2_norm: float = float('inf')
    evicted: bool = False


class ThoughtSegmenter:
    """
    Segments reasoning content into discrete thoughts using linguistic markers.

    Thought boundaries are detected using 14 target phrases that indicate a
    new line of reasoning. Token positions are computed reasoning-relative via
    tokenizer offset_mapping.

    Args:
        tokenizer: A callable tokenizer supporting return_offsets_mapping=True.
            Must be passed as a pre-loaded instance (never loads from disk).
        min_segment_tokens: Minimum tokens for a thought segment. Sub-threshold
            thoughts are merged into the next thought via greedy accumulate.
            The final thought is exempt from this threshold. Default 15.
    """

    TARGET_PHRASES = [
        "alternative", "Alternative", "Another", "But",
        "Perhaps", "perhaps another", "Wait", "Oh wait",
        "Now", "but wait", "Oh, so", "So", "In other words",
        "Similarly"
    ]

    _think_start_pattern = re.compile(r'<think>', re.IGNORECASE)
    _think_end_pattern = re.compile(r'</think>', re.IGNORECASE)

    # Characters to back up when scanning for separators that span chunk
    # boundaries.  Equal to max(len(p) for p in TARGET_PHRASES) - 1.
    _SEPARATOR_OVERLAP: int = max(len(p) for p in TARGET_PHRASES) - 1

    def __init__(self, tokenizer, min_segment_tokens: int = 15) -> None:
        self._tokenizer = tokenizer
        self._min_segment_tokens = min_segment_tokens

        self._segmentation_pattern = re.compile(
            r'(' + '|'.join(re.escape(p) for p in self.TARGET_PHRASES) + r')'
        )

        # Per-request mutable state
        self._thoughts: list[ThoughtSegment] = []
        self._processed_char_len: int = 0
        self._reasoning_content: str = ""

        # Tokenization cache: avoid re-encoding the full text every cycle.
        self._cached_offset_mapping: list[tuple[int, int]] = []
        self._cached_text_len: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def thoughts(self) -> list[ThoughtSegment]:
        """Read-only access to current thought list."""
        return self._thoughts

    def reset(self) -> None:
        """Clear all state for a new request."""
        self._thoughts = []
        self._processed_char_len = 0
        self._reasoning_content = ""
        self._cached_offset_mapping = []
        self._cached_text_len = 0

    def extract_reasoning_span(self, text: str) -> tuple[int, int] | None:
        """Extract character start/end of content between <think> and </think>.

        Args:
            text: Full generated text, potentially containing think tags.

        Returns:
            Tuple (start_char, end_char) of reasoning content inside think
            tags, or None if no complete think span is found.
        """
        start_match = self._think_start_pattern.search(text)
        if start_match is None:
            return None

        content_start = start_match.end()

        end_match = self._think_end_pattern.search(text, content_start)
        if end_match is None:
            # Think block not yet closed — return span up to current end.
            return (content_start, len(text))

        return (content_start, end_match.start())

    def update(self, reasoning_content: str) -> list[ThoughtSegment]:
        """Segment new reasoning content into thoughts.

        Processes only the text since the last call. The last existing thought
        may be extended if the new text begins with non-separator content.
        Token positions are recalculated after each update.

        Args:
            reasoning_content: Full accumulated reasoning content so far.

        Returns:
            Current list of ThoughtSegment objects (may grow between calls).
        """
        self._reasoning_content = reasoning_content
        self._segment_new_text()
        return self._thoughts

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _segment_new_text(self) -> None:
        """Segment new reasoning content since last call.

        To catch separator phrases that span chunk boundaries, the scan
        backs up ``_SEPARATOR_OVERLAP`` characters into already-processed
        text.  Thoughts covering that overlap region are trimmed or removed
        so the overlap can be re-segmented together with the new text.
        """
        text = self._reasoning_content
        if len(text) == self._processed_char_len:
            return

        # Back up to catch separators that span the chunk boundary.
        overlap = min(self._processed_char_len, self._SEPARATOR_OVERLAP)
        scan_start = self._processed_char_len - overlap

        # Trim / remove thoughts that fall within the overlap region so
        # the overlap can be re-segmented with the new text.
        if overlap > 0 and self._thoughts:
            while self._thoughts and self._thoughts[-1].start_char_pos >= scan_start:
                self._thoughts.pop()
            if self._thoughts and self._thoughts[-1].end_char_pos > scan_start:
                last = self._thoughts[-1]
                last.text = text[last.start_char_pos:scan_start]
                last.end_char_pos = scan_start
                last.start_token_pos = -1
                last.end_token_pos = -1

        scan_text = text[scan_start:]
        current_char_offset = scan_start

        parts = self._segmentation_pattern.split(scan_text)

        # First part: text before the first separator
        if parts[0]:
            if self._thoughts:
                # Extend the last existing thought
                self._thoughts[-1].text += parts[0]
                self._thoughts[-1].end_char_pos += len(parts[0])
                self._thoughts[-1].end_token_pos = -1
            else:
                self._thoughts.append(ThoughtSegment(
                    text=parts[0],
                    start_char_pos=current_char_offset,
                    end_char_pos=current_char_offset + len(parts[0]),
                    start_token_pos=-1,
                    end_token_pos=-1,
                ))
            current_char_offset += len(parts[0])

        # Remaining parts: [separator, text, separator, text, ...]
        i = 1
        while i < len(parts):
            separator = parts[i]
            text_part = parts[i + 1] if (i + 1) < len(parts) else ""
            thought_text = separator + text_part

            start_char = current_char_offset
            end_char = start_char + len(thought_text)

            self._thoughts.append(ThoughtSegment(
                text=thought_text,
                start_char_pos=start_char,
                end_char_pos=end_char,
                start_token_pos=-1,
                end_token_pos=-1,
            ))
            current_char_offset = end_char
            i += 2

        self._processed_char_len = len(text)
        self._recalculate_token_positions()
        self._merge_sub_threshold_thoughts()

    # Number of cached tokens to re-tokenize on each update to correct
    # BPE boundary effects.  The last few tokens of a prefix encoding can
    # differ from the full-text encoding because BPE merges are greedy and
    # context-dependent.  Re-tokenizing from a few tokens back ensures the
    # boundary tokens are computed with sufficient right context.
    _OVERLAP_TOKENS: int = 3

    def _recalculate_token_positions(self) -> None:
        """Calculate token positions for all thoughts using offset_mapping.

        Caches the offset_mapping from previous calls. On each update the
        last ``_OVERLAP_TOKENS`` cached tokens are discarded and the text
        from that point forward is re-tokenized. This corrects BPE boundary
        effects while keeping cost at O(overlap + delta) instead of O(n).

        Positions are relative to the start of reasoning_content.
        """
        if not self._thoughts:
            return

        text = self._reasoning_content
        text_len = len(text)

        if text_len > self._cached_text_len:
            if self._cached_offset_mapping:
                # Back up _OVERLAP_TOKENS tokens to correct boundary effects.
                overlap_idx = max(
                    0, len(self._cached_offset_mapping) - self._OVERLAP_TOKENS
                )
                retok_char_start = self._cached_offset_mapping[overlap_idx][0]
                del self._cached_offset_mapping[overlap_idx:]
            else:
                retok_char_start = 0

            encoding = self._tokenizer(
                text[retok_char_start:],
                add_special_tokens=False,
                return_offsets_mapping=True,
            )

            self._cached_offset_mapping.extend(
                (s + retok_char_start, e + retok_char_start)
                for s, e in encoding['offset_mapping']
            )
            self._cached_text_len = text_len

        offset_mapping = self._cached_offset_mapping

        for thought in self._thoughts:
            char_start = thought.start_char_pos
            char_end = thought.end_char_pos

            # Find first token whose end is past char_start
            token_start = None
            for i, (tok_char_start, tok_char_end) in enumerate(offset_mapping):
                if tok_char_end > char_start:
                    token_start = i
                    break

            # Find last token whose start is before char_end
            token_end = None
            for i in range(len(offset_mapping) - 1, -1, -1):
                tok_char_start, tok_char_end = offset_mapping[i]
                if tok_char_start < char_end:
                    token_end = i + 1  # Exclusive end
                    break

            if token_start is None:
                token_start = len(offset_mapping)
            if token_end is None:
                token_end = 0

            thought.start_token_pos = token_start
            thought.end_token_pos = max(token_start, token_end)

    def _merge_sub_threshold_thoughts(self) -> None:
        """Merge consecutive thoughts where the earlier one is below min threshold.

        Runs after _recalculate_token_positions() so token counts are valid.
        The final thought is always kept regardless of token count (D-02).
        When min_segment_tokens <= 0, no merging occurs.
        """
        if self._min_segment_tokens <= 0 or len(self._thoughts) < 2:
            return

        merged: list[ThoughtSegment] = []
        for thought in self._thoughts:
            if (merged
                    and (merged[-1].end_token_pos - merged[-1].start_token_pos)
                        < self._min_segment_tokens):
                # Extend previous thought to absorb this one
                prev = merged[-1]
                prev.text += thought.text
                prev.end_char_pos = thought.end_char_pos
                prev.end_token_pos = thought.end_token_pos
            else:
                merged.append(thought)
        self._thoughts = merged
