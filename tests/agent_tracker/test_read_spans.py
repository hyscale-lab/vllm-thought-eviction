from vllm.agent_tracker.classifier import TOOL_FILE_READ
from vllm.agent_tracker.read_spans import (
    FileReadSpan,
    extract_read_spans_from_messages,
)
from vllm.agent_tracker.segment_map import TurnState
from vllm.agent_tracker.tracker import SessionTracker


def _bash_call(tool_call_id: str, command: str) -> dict:
    return {
        "role": "assistant",
        "tool_calls": [{
            "id": tool_call_id,
            "function": {
                "name": "bash",
                "arguments": {"command": command},
            },
        }],
    }


def _tool_msg(tool_call_id: str, content: str = "") -> dict:
    return {"role": "tool", "tool_call_id": tool_call_id, "content": content}


def _turn(turn_idx: int, span: FileReadSpan) -> TurnState:
    return TurnState(
        turn_idx=turn_idx,
        category=TOOL_FILE_READ,
        files_referenced={span.path.split("/")[-1]},
        files_referenced_full={span.path},
        read_spans=[span],
        token_count=10,
        observation_tokens=10,
        obs_token_range=(turn_idx * 10, turn_idx * 10 + 10),
        round_idx=turn_idx + 1,
    )


def test_extracts_exact_sed_read_span() -> None:
    assistant = _bash_call("call_1", "sed -n '10,25p' '/app/pkg/mod.py'")
    spans = extract_read_spans_from_messages([_tool_msg("call_1")], [assistant])

    assert spans == [FileReadSpan(
        path="/app/pkg/mod.py",
        start_line=10,
        end_line=25,
        kind="line_range",
        confidence="exact",
    )]


def test_extracts_grep_match_line_spans() -> None:
    assistant = _bash_call(
        "call_1",
        "grep -rnH --exclude-dir='.*' 'needle' '/app/pkg' | head -n 50",
    )
    output = "/app/pkg/a.py:41:def needle():\n/app/pkg/a.py:52:needle()"
    spans = extract_read_spans_from_messages([_tool_msg("call_1", output)], [assistant])

    assert spans == [
        FileReadSpan("/app/pkg/a.py", 41, 41, "line_match", "exact"),
        FileReadSpan("/app/pkg/a.py", 52, 52, "line_match", "exact"),
    ]


def test_non_overlapping_same_file_read_does_not_supersede() -> None:
    tracker = SessionTracker("span-test", n_decay=999)
    prior = _turn(0, FileReadSpan("/app/a.py", 1, 50, "line_range", "exact"))
    later = _turn(1, FileReadSpan("/app/a.py", 100, 150, "line_range", "exact"))
    tracker.evictable_map.append(prior)
    tracker.evictable_map.append(later)

    tracker._reclassify_priors_for_new_turn(1)

    assert not prior.evictable
    assert prior.eviction_reason == "essential"
    assert prior.superseded_by is None


def test_covering_later_read_supersedes_prior() -> None:
    tracker = SessionTracker("span-test", n_decay=999)
    prior = _turn(0, FileReadSpan("/app/a.py", 20, 30, "line_range", "exact"))
    later = _turn(1, FileReadSpan("/app/a.py", 1, 100, "line_range", "exact"))
    tracker.evictable_map.append(prior)
    tracker.evictable_map.append(later)

    tracker._reclassify_priors_for_new_turn(1)

    assert prior.evictable
    assert prior.eviction_reason == "superseded_by_later_read"
    assert prior.superseded_by == 1


def test_partial_overlap_does_not_evict_whole_prior_observation() -> None:
    tracker = SessionTracker("span-test", n_decay=999)
    prior = _turn(0, FileReadSpan("/app/a.py", 20, 40, "line_range", "exact"))
    later = _turn(1, FileReadSpan("/app/a.py", 30, 50, "line_range", "exact"))
    tracker.evictable_map.append(prior)
    tracker.evictable_map.append(later)

    tracker._reclassify_priors_for_new_turn(1)

    assert not prior.evictable


def test_later_full_file_read_supersedes_prior_span() -> None:
    tracker = SessionTracker("span-test", n_decay=999)
    prior = _turn(0, FileReadSpan("/app/a.py", 20, 40, "line_range", "exact"))
    later = _turn(1, FileReadSpan("/app/a.py", kind="full_file", confidence="full"))
    tracker.evictable_map.append(prior)
    tracker.evictable_map.append(later)

    tracker._reclassify_priors_for_new_turn(1)

    assert prior.evictable
    assert prior.eviction_reason == "superseded_by_later_read"


def test_span_paths_can_supersede_when_file_reference_was_directory() -> None:
    tracker = SessionTracker("span-test", n_decay=999)
    prior = _turn(0, FileReadSpan("/app/pkg/a.py", 41, 41, "line_match", "exact"))
    prior.files_referenced = {"pkg"}
    prior.files_referenced_full = {"/app/pkg"}
    later = _turn(1, FileReadSpan("/app/pkg/a.py", 1, 100, "line_range", "exact"))
    tracker.evictable_map.append(prior)
    tracker.evictable_map.append(later)

    tracker._reclassify_priors_for_new_turn(1)

    assert prior.evictable
    assert prior.eviction_reason == "superseded_by_later_read"


def test_all_prior_spans_must_be_covered_before_whole_observation_eviction() -> None:
    tracker = SessionTracker("span-test", n_decay=999)
    prior = _turn(0, FileReadSpan("/app/a.py", 20, 20, "line_match", "exact"))
    prior.read_spans.append(FileReadSpan("/app/a.py", 80, 80, "line_match", "exact"))
    later = _turn(1, FileReadSpan("/app/a.py", 1, 50, "line_range", "exact"))
    tracker.evictable_map.append(prior)
    tracker.evictable_map.append(later)

    tracker._reclassify_priors_for_new_turn(1)

    assert not prior.evictable


def test_narrower_read_knob_is_required_for_subset_eviction() -> None:
    prior = _turn(0, FileReadSpan("/app/a.py", 1, 200, "line_range", "exact"))
    later = _turn(1, FileReadSpan("/app/a.py", 50, 80, "line_range", "exact"))

    tracker = SessionTracker("span-test", n_decay=999)
    tracker.evictable_map.append(prior)
    tracker.evictable_map.append(later)
    tracker._reclassify_priors_for_new_turn(1)
    assert not prior.evictable


def test_narrower_read_knob_evicts_broader_single_span_prior() -> None:
    tracker = SessionTracker("span-test", n_decay=999, evict_narrower_reads=True)
    prior = _turn(0, FileReadSpan("/app/a.py", 1, 200, "line_range", "exact"))
    later = _turn(1, FileReadSpan("/app/a.py", 50, 80, "line_range", "exact"))
    tracker.evictable_map.append(prior)
    tracker.evictable_map.append(later)

    tracker._reclassify_priors_for_new_turn(1)

    assert prior.evictable
    assert prior.eviction_reason == "superseded_by_narrower_read"
    assert prior.superseded_by == 1


def test_narrower_read_knob_evicts_full_file_prior() -> None:
    tracker = SessionTracker("span-test", n_decay=999, evict_narrower_reads=True)
    prior = _turn(0, FileReadSpan("/app/a.py", kind="full_file", confidence="full"))
    later = _turn(1, FileReadSpan("/app/a.py", 50, 80, "line_range", "exact"))
    tracker.evictable_map.append(prior)
    tracker.evictable_map.append(later)

    tracker._reclassify_priors_for_new_turn(1)

    assert prior.evictable
    assert prior.eviction_reason == "superseded_by_narrower_read"


def test_narrower_read_knob_does_not_evict_multi_span_prior() -> None:
    tracker = SessionTracker("span-test", n_decay=999, evict_narrower_reads=True)
    prior = _turn(0, FileReadSpan("/app/a.py", 1, 100, "line_range", "exact"))
    prior.read_spans.append(FileReadSpan("/app/a.py", 200, 300, "line_range", "exact"))
    later = _turn(1, FileReadSpan("/app/a.py", 20, 30, "line_range", "exact"))
    tracker.evictable_map.append(prior)
    tracker.evictable_map.append(later)

    tracker._reclassify_priors_for_new_turn(1)

    assert not prior.evictable
