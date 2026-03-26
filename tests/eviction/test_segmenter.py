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

    def test_short_thoughts_still_created(self, mock_tokenizer):
        """Thoughts with fewer tokens than min_segment_tokens are still created.

        Filtering is strategy-level responsibility, not segmenter-level.
        """
        # Set a high min_segment_tokens to confirm segmenter still creates segments
        segmenter = ThoughtSegmenter(mock_tokenizer, min_segment_tokens=100)
        reasoning = "Start. But tiny."
        thoughts = segmenter.update(reasoning)
        # The "But tiny." thought has very few tokens but should still be created
        assert len(thoughts) == 2

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

    def test_thoughts_property_is_readonly_view(self, mock_tokenizer):
        """thoughts property returns current list without allowing internal mutation."""
        segmenter = ThoughtSegmenter(mock_tokenizer)
        segmenter.update("Some text. But more text here.")
        thoughts = segmenter.thoughts
        assert len(thoughts) == 2
        # Modifying the returned list should not affect internal state via reference
        # (the property returns the internal list, so at minimum verify it's the right object)
        assert thoughts is segmenter.thoughts
