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
    # D-09: per-session N-decay tail-guard window (rounds). When set, overrides
    # the SessionTracker default (3). Higher = more conservative (spares more
    # recent rounds before decay-evicting). Swept per-arm via the client's
    # AGENT_TRACKER_N_DECAY knob; None keeps the server default. The upper bound
    # is intentionally large so an arm can set n_decay >> trajectory length
    # (e.g. 999) to effectively DISABLE time-decay and evict purely on
    # supersession (superseded_by_later_read / _by_edit / _by_repeat).
    n_decay: int | None = Field(default=None, ge=1, le=9999)
    # Ablation: also drop the paired assistant "Agent tool call" turn when every
    # tool-result turn in its round is evicted (serving.py Pass 2). Off by
    # default so the n_decay/supersession arms drop only tool OUTPUT tokens; the
    # `droptc` arm sets it to measure the marginal effect of reclaiming the call.
    evict_tool_call: bool = Field(default=False)
    # Ablation: content-hash dedupe of repeated command output (run/exec +
    # other-bash). When set, a later turn whose normalized observation text
    # matches an earlier such turn supersedes it (reason superseded_by_repeat).
    # Per-session, locked at tracker creation like n_decay.
    dedupe_cmd_output: bool = Field(default=False)
    # Ablation: epoch-batched (cache-friendly) eviction. When set to K>1, NEW
    # evictions are committed to the engine prompt only on requests whose round
    # sits on an epoch boundary (round % K == 0); turns already dropped stay
    # dropped (monotone). Hard eviction invalidates the prefix cache at the
    # earliest newly-dropped span every time the drop set changes -- measured at
    # 50-62% of requests on the n5/n10 arms, inflating actual prefill compute
    # 1.3-4.3x over noevict. Batching the drop-set changes cuts that
    # invalidation frequency to ~1/K with the same steady-state token savings.
    # Per-request gate applied in serving.py Pass 1; None/1 = evict every
    # request (legacy behavior).
    evict_epoch: int | None = Field(default=None, ge=1, le=9999)


class TurnOpportunity(OpenAIBaseModel):
    """One row of the opportunity-endpoint response (D-16)."""
    turn_idx: int
    category: str
    evictable: bool
    reason: str  # essential | superseded_by_edit | superseded_by_later_read | superseded_by_repeat | superseded_by_fetch | superseded_by_new_search | decayed_N_turns
    msg_range: tuple[int, int]
    token_range: tuple[int, int]
    obs_token_range: tuple[int, int] | None = None
    superseded_by: int | None = None
    # Agent round this turn belongs to (one request ~ one round).
    round_idx: int = 0
    # Round at which server-side eviction FIRST dropped this turn from the
    # prefill; None if it was never actually evicted (e.g. reported as an
    # opportunity but eviction disabled, or still inside the no-evict zone).
    evicted_at_round: int | None = None


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
