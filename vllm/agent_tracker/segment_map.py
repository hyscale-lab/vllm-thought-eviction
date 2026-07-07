"""TurnState dataclass and EvictableSegmentMap (D-07, second indexed view).

TurnState is ported from `scripts/trajectory_classifier.py:109-126` and augmented
with three new fields (msg_range, token_range, obs_token_hash) per RESEARCH
section 5. EvictableSegmentMap is the primary opportunity map Phase 2 will read.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TurnState:
    """Per-turn state. Mirrors `scripts.trajectory_classifier.TurnState` plus
    three new fields needed by the live tracker (D-07/D-11)."""
    turn_idx: int
    category: str
    files_referenced: set[str] = field(default_factory=set)
    files_referenced_full: set[str] = field(default_factory=set)
    # Normalized URLs this turn references: search-result links for a
    # TOOL_WEB_SEARCH turn, the fetch target for a TOOL_WEB_FETCH turn. Empty
    # for every other category. Drives web supersession (see
    # SessionTracker._reclassify_priors_for_new_web_turn).
    urls_referenced: set[str] = field(default_factory=set)
    command: str | None = None
    is_edit: bool = False
    is_success: bool = True
    token_count: int = 0
    observation_tokens: int = 0
    reasoning_tokens: int = 0
    tool_call_tokens: int = 0
    other_tokens: int = 0
    evictable: bool = False
    eviction_reason: str = "essential"
    superseded_by: int | None = None
    # NEW for live tracker (RESEARCH section 5):
    msg_range: tuple[int, int] = (0, 0)        # [start_msg_idx, end_msg_idx)
    token_range: tuple[int, int] = (0, 0)      # [start_token_idx, end_token_idx)
    obs_token_range: tuple[int, int] | None = None # [start_token_idx, end_token_idx) for observation message
    obs_token_hash: bytes | None = None        # blake2b digest of observation tokens (D-11)
    obs_norm_hash: bytes | None = None          # blake2b digest of NORMALIZED obs text, run/exec + other-bash only (content-hash dedupe)
    # Per-tool eviction: the assistant-anchored ROUND this turn
    # One agent action (assistant completion + its tool results) =
    # one round; N-decay counts rounds, not raw turns, so a burst of parallel
    # tool turns doesn't exhaust the lookahead window.
    round_idx: int = 0
    # The ROUND at which this turn's tokens were FIRST dropped from the engine
    # prefill by server-side eviction (serving.py). None until actually evicted;
    # first-drop wins (re-dropped every subsequent request, but the round is
    # recorded once). Surfaced in the opportunity JSON so offline analysis can
    # see WHEN a section left the context, not just that it was evictable.
    evicted_at_round: int | None = None


class EvictableSegmentMap:
    """`turn_idx -> TurnState` view (D-07).

    Thin wrapper around a `list[TurnState]` indexed by turn_idx. The
    SessionTracker mutates `evictable`, `eviction_reason`, `superseded_by`
    on existing entries when new evidence arrives (D-08).
    """

    def __init__(self) -> None:
        self._turns: list[TurnState] = []

    def __len__(self) -> int:
        return len(self._turns)

    def __iter__(self):
        return iter(self._turns)

    def __getitem__(self, turn_idx: int) -> TurnState:
        return self._turns[turn_idx]

    def append(self, ts: TurnState) -> None:
        assert ts.turn_idx == len(self._turns), (
            f"TurnState.turn_idx={ts.turn_idx} must equal next slot "
            f"{len(self._turns)}"
        )
        self._turns.append(ts)

    def all_turns(self) -> list[TurnState]:
        return self._turns
