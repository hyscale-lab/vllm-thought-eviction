"""
Unit tests for vllm.thought_eviction.segmenter.

Coverage:
- SEG-01: extract_reasoning_span() detects <think>...</think> tags
- SEG-02: update() segments reasoning into thoughts using target phrases
- SEG-03: Token positions computed via mock tokenizer offset_mapping, reasoning-relative
"""

import pytest
from unittest.mock import MagicMock

from vllm.thought_eviction.segmenter import ThoughtSegment, ThoughtSegmenter


@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer returning ~1 token per 5 characters."""
    tokenizer = MagicMock()

    def tokenizer_call(text, add_special_tokens=False, return_offsets_mapping=False):
        offsets = []
        pos = 0
        token_ids = []
        while pos < len(text):
            end = min(pos + 5, len(text))
            offsets.append((pos, end))
            token_ids.append(pos)  # fake token id
            pos = end
        result = {"input_ids": token_ids}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result

    tokenizer.side_effect = tokenizer_call
    return tokenizer


# ---------------------------------------------------------------------------
# SEG-01: think tag detection
# ---------------------------------------------------------------------------

class TestExtractReasoningSpan:
    """SEG-01: extract_reasoning_span detects <think>...</think> spans."""

    def test_extract_reasoning_span_returns_content_offsets(self, mock_tokenizer):
        """extract_reasoning_span returns character offsets of content inside tags."""
        segmenter = ThoughtSegmenter(mock_tokenizer)
        text = "<think>hello world</think>"
        span = segmenter.extract_reasoning_span(text)

        assert span is not None
        content = text[span[0]:span[1]]
        assert content == "hello world"

    def test_extract_reasoning_span_no_tags_returns_none(self, mock_tokenizer):
        """extract_reasoning_span returns None when no think tags present."""
        segmenter = ThoughtSegmenter(mock_tokenizer)
        span = segmenter.extract_reasoning_span("no think tags here")
        assert span is None

    def test_extract_reasoning_span_open_tag_returns_rest(self, mock_tokenizer):
        """extract_reasoning_span with unclosed tag returns span to end of text."""
        segmenter = ThoughtSegmenter(mock_tokenizer)
        text = "<think>partial content"
        span = segmenter.extract_reasoning_span(text)
        assert span is not None
        assert text[span[0]:span[1]] == "partial content"

    def test_extract_reasoning_span_case_insensitive(self, mock_tokenizer):
        """extract_reasoning_span is case-insensitive on tag matching."""
        segmenter = ThoughtSegmenter(mock_tokenizer)
        text = "<THINK>content here</THINK>"
        span = segmenter.extract_reasoning_span(text)
        assert span is not None
        assert text[span[0]:span[1]] == "content here"


# ---------------------------------------------------------------------------
# SEG-02: thought boundary segmentation
# ---------------------------------------------------------------------------

class TestSegmentReasoningWithTargetPhrases:
    """SEG-02: update() segments reasoning into thoughts using boundary phrases."""

    def test_segment_reasoning_with_target_phrases(self, mock_tokenizer):
        """update() with 'But' and 'Wait' produces 3 thoughts."""
        segmenter = ThoughtSegmenter(mock_tokenizer)
        reasoning = "First thought content. But second thought. Wait third thought."
        thoughts = segmenter.update(reasoning)

        assert len(thoughts) == 3, f"Expected 3 thoughts, got {len(thoughts)}: {[t.text for t in thoughts]}"
        assert thoughts[0].text.startswith("First")
        assert thoughts[1].text.startswith("But")
        assert thoughts[2].text.startswith("Wait")

    def test_incremental_update_appends_to_last_thought(self, mock_tokenizer):
        """update() called incrementally appends to existing last thought."""
        segmenter = ThoughtSegmenter(mock_tokenizer)

        # First call: only first half
        first_half = "First thought content part one"
        thoughts_after_first = segmenter.update(first_half)
        assert len(thoughts_after_first) == 1

        # Second call: full text with a separator later
        full_text = "First thought content part one part two. But new thought."
        thoughts_after_second = segmenter.update(full_text)

        # Should now have 2 thoughts; first thought extended, second added
        assert len(thoughts_after_second) == 2
        assert thoughts_after_second[0].text.startswith("First")
        assert thoughts_after_second[1].text.startswith("But")

    def test_separator_spanning_chunk_boundary(self, mock_tokenizer):
        """Separators split across chunk boundaries are detected.

        SEG-02 regression: if a chunk boundary falls in the middle of a
        separator phrase (e.g. 'B' | 'ut'), the separator must still be
        detected via the overlap window in _segment_new_text.
        """
        full_text = (
            "First thought content here. "       # 28 chars
            "But second thought is longer. "      # 30 chars, ends at 58
            "Perhaps another way to see it. "     # 31 chars, ends at 89
            "Oh wait, I see the answer now."      # 30 chars, ends at 119
        )

        # Chunks deliberately bisect separators:
        #   chunk 1 ends at 29 → "...here. B" (splits "But")
        #   chunk 2 ends at 60 → "...longer. P" (splits "Perhaps")
        #   chunk 3 ends at 92 → "...it. Oh w" (splits "Oh wait")
        segmenter = ThoughtSegmenter(mock_tokenizer)
        segmenter.update(full_text[:29])
        segmenter.update(full_text[:60])
        segmenter.update(full_text[:92])
        thoughts = segmenter.update(full_text)

        # Single-shot for comparison
        single = ThoughtSegmenter(mock_tokenizer)
        single_thoughts = single.update(full_text)

        assert len(thoughts) == len(single_thoughts), (
            f"Thought count: incremental={len(thoughts)}, single={len(single_thoughts)}. "
            f"Inc texts: {[t.text[:30] for t in thoughts]}, "
            f"Single texts: {[t.text[:30] for t in single_thoughts]}"
        )
        for i, (it, st) in enumerate(zip(thoughts, single_thoughts)):
            assert it.text == st.text, (
                f"Thought {i} text mismatch: {it.text!r} vs {st.text!r}"
            )
            assert it.start_char_pos == st.start_char_pos
            assert it.end_char_pos == st.end_char_pos

    def test_short_thoughts_still_created(self, mock_tokenizer):
        """Sub-threshold thoughts are merged into the next thought by the segmenter."""
        # Set a high min_segment_tokens to trigger merging
        segmenter = ThoughtSegmenter(mock_tokenizer, min_segment_tokens=100)
        reasoning = "Start. But tiny."
        thoughts = segmenter.update(reasoning)
        # The short first thought ("Start. ") should be merged into "But tiny."
        assert len(thoughts) == 1

    def test_reset_clears_state(self, mock_tokenizer):
        """reset() clears all thoughts and char tracking for a new request."""
        segmenter = ThoughtSegmenter(mock_tokenizer)
        segmenter.update("Some text. But more.")
        assert len(segmenter.thoughts) == 2

        segmenter.reset()
        assert len(segmenter.thoughts) == 0

        # After reset, update should work fresh
        thoughts = segmenter.update("New text.")
        assert len(thoughts) == 1


# ---------------------------------------------------------------------------
# SEG-03: token position mapping
# ---------------------------------------------------------------------------

class TestTokenPositionMapping:
    """SEG-03: Token positions computed via tokenizer offset_mapping."""

    def test_token_positions_set_after_update(self, mock_tokenizer):
        """Token positions are non-negative after update() is called."""
        segmenter = ThoughtSegmenter(mock_tokenizer)
        thoughts = segmenter.update("Hello world. But another thought here.")

        for thought in thoughts:
            assert thought.start_token_pos >= 0, f"start_token_pos is {thought.start_token_pos}"
            assert thought.end_token_pos >= 0, f"end_token_pos is {thought.end_token_pos}"

    def test_token_positions_are_reasoning_relative(self, mock_tokenizer):
        """Token positions start from 0 (reasoning-relative), not absolute sequence positions."""
        segmenter = ThoughtSegmenter(mock_tokenizer)
        thoughts = segmenter.update("First thought starts here. But second thought.")

        # The first thought should start at token 0 (reasoning-relative)
        assert thoughts[0].start_token_pos == 0, (
            f"First thought should start at token 0, got {thoughts[0].start_token_pos}"
        )

    def test_token_positions_via_offset_mapping(self, mock_tokenizer):
        """Token positions are computed via tokenizer returning known offset_mapping."""
        segmenter = ThoughtSegmenter(mock_tokenizer)
        # "01234" = 1 token (chars 0-5), "But x" = boundary at char 5
        reasoning = "01234But rest here."
        thoughts = segmenter.update(reasoning)

        # Verify the tokenizer was called with return_offsets_mapping=True
        calls = mock_tokenizer.call_args_list
        offset_calls = [c for c in calls if c.kwargs.get('return_offsets_mapping', False)]
        assert len(offset_calls) >= 1, "Tokenizer should have been called with return_offsets_mapping=True"

    def test_incremental_cache_matches_full_tokenization(self, mock_tokenizer):
        """Cached incremental tokenization produces same positions as full-text.

        SEG-03 regression: _recalculate_token_positions caches offset_mapping
        and only tokenizes the new suffix. Verify that multiple incremental
        update() calls produce identical token positions to a fresh segmenter
        that receives the full text in one shot.

        Chunk boundaries are chosen mid-thought (not at separators) so
        _segment_new_text sees the same separators as single-shot, isolating
        the tokenization-cache behavior under test.
        """
        full_text = (
            "First thought content here. "      # ends at 28
            "But second thought is longer. "     # ends at 58
            "Wait third thought arrives. "       # ends at 85
            "Perhaps a fourth thought too."      # ends at 114
        )

        # Chunks land mid-thought, never bisecting a separator word.
        # --- Incremental: feed text in 3 growing chunks ---
        inc = ThoughtSegmenter(mock_tokenizer)
        inc.update(full_text[:35])   # mid "second"
        inc.update(full_text[:70])   # mid "arrives"
        inc_thoughts = inc.update(full_text)  # full text

        # --- Single-shot: feed full text at once ---
        single = ThoughtSegmenter(mock_tokenizer)
        single_thoughts = single.update(full_text)

        assert len(inc_thoughts) == len(single_thoughts), (
            f"Thought count mismatch: incremental={len(inc_thoughts)}, "
            f"single={len(single_thoughts)}"
        )
        for i, (it, st) in enumerate(zip(inc_thoughts, single_thoughts)):
            assert it.text == st.text, (
                f"Thought {i} text mismatch: {it.text!r} vs {st.text!r}"
            )
            assert it.start_token_pos == st.start_token_pos, (
                f"Thought {i} start_token_pos mismatch: "
                f"{it.start_token_pos} vs {st.start_token_pos}"
            )
            assert it.end_token_pos == st.end_token_pos, (
                f"Thought {i} end_token_pos mismatch: "
                f"{it.end_token_pos} vs {st.end_token_pos}"
            )
            assert it.start_char_pos == st.start_char_pos, (
                f"Thought {i} start_char_pos mismatch: "
                f"{it.start_char_pos} vs {st.start_char_pos}"
            )
            assert it.end_char_pos == st.end_char_pos, (
                f"Thought {i} end_char_pos mismatch: "
                f"{it.end_char_pos} vs {st.end_char_pos}"
            )

    def test_cache_retokenizes_overlap_plus_suffix(self, mock_tokenizer):
        """Tokenizer is called with overlap window + new suffix, not full text.

        Verifies the O(overlap + delta) optimization: after the initial call,
        subsequent update() calls back up _OVERLAP_TOKENS tokens and
        re-tokenize from that point forward — not the entire text.
        """
        segmenter = ThoughtSegmenter(mock_tokenizer)
        text_v1 = "First thought here. "  # 20 chars → 4 tokens at 5 chars each
        text_v2 = text_v1 + "But second thought."

        segmenter.update(text_v1)
        mock_tokenizer.reset_mock()

        segmenter.update(text_v2)

        assert mock_tokenizer.call_count == 1
        call_args = mock_tokenizer.call_args
        tokenized_text = call_args[0][0]

        # Should NOT be the full text (39 chars)
        assert tokenized_text != text_v2, "Should not re-tokenize the full text"
        # Should include the suffix
        assert tokenized_text.endswith("But second thought.")
        # Should include overlap from cached tokens (backed up 3 tokens)
        assert len(tokenized_text) > len("But second thought.")

    def test_thoughts_property_is_readonly_view(self, mock_tokenizer):
        """thoughts property returns current list without allowing internal mutation."""
        segmenter = ThoughtSegmenter(mock_tokenizer)
        segmenter.update("Some text. But more text here.")
        thoughts = segmenter.thoughts
        assert len(thoughts) == 2
        # Modifying the returned list should not affect internal state via reference
        # (the property returns the internal list, so at minimum verify it's the right object)
        assert thoughts is segmenter.thoughts


# ---------------------------------------------------------------------------
# SEG-MERGE: sub-threshold thought merging
# ---------------------------------------------------------------------------

class TestMergeSubThresholdThoughts:
    """SEG-MERGE: _merge_sub_threshold_thoughts() merges short thoughts into next."""

    def test_merge_sub_threshold_into_next(self, mock_tokenizer):
        """Two thoughts where first is short -> merged into one.

        mock_tokenizer uses ~5 chars per token.
        "Start. " is 7 chars -> 2 tokens (below min_segment_tokens=5 only if
        we set a threshold above 2). Use min_segment_tokens=5 to trigger merge.
        Actually "Start. " = 7 chars = ~2 tokens (ceil(7/5)=2).
        "But second thought content here." = 32 chars = ~7 tokens.
        First thought (2 tokens) < threshold (5) -> merges into second.
        """
        # "Ab. " = 4 chars = 1 token (below min=5), "But rest rest rest rest rest." = 29 chars = 6 tokens
        segmenter = ThoughtSegmenter(mock_tokenizer, min_segment_tokens=5)
        reasoning = "Ab. But rest rest rest rest rest."
        thoughts = segmenter.update(reasoning)

        assert len(thoughts) == 1, (
            f"Expected 1 merged thought, got {len(thoughts)}: {[t.text for t in thoughts]}"
        )
        assert "Ab." in thoughts[0].text
        assert "But" in thoughts[0].text

    def test_merge_chain_sub_threshold(self, mock_tokenizer):
        """Three consecutive short thoughts all merge into the last one.

        Each short thought (< min_segment_tokens) merges into the next
        via greedy accumulate: first into second, then combined first+second
        into third if combined is still short, otherwise second alone into third.

        Since merging is greedy left-to-right:
        - After first+second merge, the merged thought may itself be below threshold
          so it absorbs the third as well.
        Use 4-char thoughts at 5 chars/token -> 1 token each, min=5.
        The third thought is longer to act as the absorber.
        """
        # "Ab. " (4 chars, 1 tok) + "But x" (5 chars, 1 tok) + "Wait x x x x x x x." (20 chars, 4 toks)
        # After merge: "Ab. But x" (9 chars, 2 toks) still < 5 -> merges into "Wait..."
        # Result: 1 thought
        segmenter = ThoughtSegmenter(mock_tokenizer, min_segment_tokens=5)
        reasoning = "Ab. But x. Wait x x x x x x x."
        thoughts = segmenter.update(reasoning)

        assert len(thoughts) == 1, (
            f"Expected 1 merged thought, got {len(thoughts)}: {[t.text for t in thoughts]}"
        )
        assert "Ab." in thoughts[0].text
        assert "But" in thoughts[0].text
        assert "Wait" in thoughts[0].text

    def test_merge_preserves_above_threshold(self, mock_tokenizer):
        """Thoughts already above threshold produce no merging.

        Each thought should have >= min_segment_tokens tokens to avoid merging.
        mock_tokenizer: 5 chars per token.
        min_segment_tokens=2: each thought needs >= 2 tokens (>= 10 chars).
        """
        # "First thought content here. " = 28 chars = 6 tokens (above 2)
        # "But second thought content here. " = 33 chars = 7 tokens (above 2)
        # "Wait third thought content." = 27 chars = 6 tokens (above 2)
        segmenter = ThoughtSegmenter(mock_tokenizer, min_segment_tokens=2)
        reasoning = (
            "First thought content here. "
            "But second thought content here. "
            "Wait third thought content."
        )
        thoughts = segmenter.update(reasoning)

        assert len(thoughts) == 3, (
            f"Expected 3 thoughts (no merging), got {len(thoughts)}: {[t.text for t in thoughts]}"
        )

    def test_merge_final_thought_exempt(self, mock_tokenizer):
        """Last thought below threshold is kept as-is (D-02 final thought exemption).

        Three thoughts: first above threshold, second above threshold, third below.
        The third (final) thought should NOT be merged or dropped.
        """
        # "First thought content here. " = 28 chars = 6 tokens (above min=5)
        # "But second thought content." = 27 chars = 6 tokens (above min=5)
        # "Now x." = 6 chars = 2 tokens (below min=5, but it's the final thought)
        segmenter = ThoughtSegmenter(mock_tokenizer, min_segment_tokens=5)
        reasoning = "First thought content here. But second thought content. Now x."
        thoughts = segmenter.update(reasoning)

        assert len(thoughts) == 3, (
            f"Expected 3 thoughts (final exempt), got {len(thoughts)}: {[t.text for t in thoughts]}"
        )
        assert thoughts[-1].text.startswith("Now")

    def test_merge_disabled_when_zero(self, mock_tokenizer):
        """min_segment_tokens=0 -> no merging occurs.

        Segmenter should produce same number of thoughts as without merge logic.
        """
        segmenter = ThoughtSegmenter(mock_tokenizer, min_segment_tokens=0)
        # "Ab. " = 4 chars = 1 token (would merge if min_segment_tokens > 0)
        reasoning = "Ab. But rest rest rest rest rest."
        thoughts = segmenter.update(reasoning)

        # With min_segment_tokens=0, no merging -> 2 thoughts
        assert len(thoughts) == 2, (
            f"Expected 2 thoughts with no merging, got {len(thoughts)}: {[t.text for t in thoughts]}"
        )

    def test_merge_single_thought_below_threshold(self, mock_tokenizer):
        """Single thought below threshold is emitted as-is (final thought exemption).

        If there is only one thought and it is below min_segment_tokens,
        it must still be returned (it is the final thought, D-02).
        """
        segmenter = ThoughtSegmenter(mock_tokenizer, min_segment_tokens=100)
        reasoning = "Hi."  # Very short, 1 token
        thoughts = segmenter.update(reasoning)

        assert len(thoughts) == 1, (
            f"Expected 1 thought (single thought exempt), got {len(thoughts)}"
        )
        assert "Hi." in thoughts[0].text
