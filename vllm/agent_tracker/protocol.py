"""Pydantic v2 schemas for the Phase 01.4 live trajectory tracker.

AgentTrackerParams is the per-request gate (D-04). The other three classes
define the GET /v1/agent_tracker/sessions/{id}/opportunity response (D-16).
"""
from __future__ import annotations

from pydantic import Field

from vllm.entrypoints.openai.engine.protocol import OpenAIBaseModel


class AgentTrackerParams(OpenAIBaseModel):
    """Phase 01.4 live trajectory tracker request params (D-04).

    When set on ChatCompletionRequest, the server maintains per-session
    eviction-opportunity state. INDEPENDENT of eviction_params --
    setting agent_tracker alone does NOT enable real KV-cache eviction.
    """
    session_id: str
    enabled: bool = True
    server_side_prompt_eviction: bool = Field(
        default=False,
        description=(
            "If true, the server actively drops evictable turns from the "
            "engine's prompt_token_ids before generation."
        ),
    )
    # D-09: reserved for future configurability; not exposed in v1.
    # n_decay: int | None = Field(default=None, ge=1, le=20)


class TurnOpportunity(OpenAIBaseModel):
    """One row of the opportunity-endpoint response (D-16)."""
    turn_idx: int
    category: str
    evictable: bool
    reason: str  # essential | superseded_by_edit | superseded_by_later_read | decayed_N_turns
    msg_range: tuple[int, int]
    token_range: tuple[int, int]
    obs_token_range: tuple[int, int] | None = None
    superseded_by: int | None = None


class ExactMatchHit(OpenAIBaseModel):
    """One token-sequence exact-match hit between observation messages (D-07/D-11)."""
    turn_idx: int
    matches_turn_idx: int
    token_count: int


class OpportunityResponse(OpenAIBaseModel):
    """D-16 endpoint response shape for GET /v1/agent_tracker/sessions/{id}/opportunity."""
    session_id: str
    n_turns: int
    # D-10: dynamic per session. Computed on the session's FIRST request as the
    # sum of token-range lengths of leading role=system messages plus the FIRST
    # role=user message; locked for the rest of the session. NOT a hardcoded
    # constant -- the legacy NO_EVICT_ZONE_P95=1991 from Phase 01.1 is intentionally
    # not used by the live tracker.
    no_evict_zone_tokens: int = Field(
        ...,
        description=(
            "Per-session no-evict zone in tokens (D-10). Dynamically computed at "
            "session start from the leading system + first user message; not a "
            "hardcoded constant."
        ),
    )
    evictable_token_total: int
    evictable_pct_of_total: float
    exact_match_turns: list[ExactMatchHit]
    turns: list[TurnOpportunity]
    file_timeline: dict[str, list[dict]]
