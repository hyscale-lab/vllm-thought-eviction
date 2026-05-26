# SPDX-License-Identifier: Apache-2.0
"""Prometheus metrics for agent-tracker server-side prompt eviction.

These are simple, label-less Counters registered against prometheus_client's
default registry, which is what vllm's /metrics endpoint exposes. They live
in the API-server frontend process where the eviction filter runs, so they
do not need the per-engine plumbing used by vllm/v1/metrics/loggers.py.

Counters:
  vllm:agent_eviction_prefill_tokens_raw_total
      Cumulative prompt-token count seen by the agent-tracker eviction hook
      BEFORE filtering (i.e., what the client sent for prefill). Incremented
      on every chat-completion request that exercises the eviction path,
      regardless of whether any tokens were actually dropped.

  vllm:agent_eviction_tokens_filtered_total
      Cumulative number of prompt tokens that the eviction hook removed
      before sending the prefill to the engine. Difference between
      raw_total and this counter equals the post-filtering prefill volume.

  vllm:agent_eviction_filter_invocations_total
      Cumulative count of requests where the filter actually dropped at
      least one token. Useful to confirm the eviction path is firing at all.
"""
from __future__ import annotations

from prometheus_client import Counter

_AGENT_EVICTION_PREFILL_TOKENS_RAW = Counter(
    "vllm:agent_eviction_prefill_tokens_raw_total",
    "Cumulative prompt tokens seen by the agent-tracker eviction hook "
    "before filtering.",
)

_AGENT_EVICTION_TOKENS_FILTERED = Counter(
    "vllm:agent_eviction_tokens_filtered_total",
    "Cumulative prompt tokens removed by the agent-tracker eviction "
    "filter before prefill.",
)

_AGENT_EVICTION_FILTER_INVOCATIONS = Counter(
    "vllm:agent_eviction_filter_invocations_total",
    "Cumulative number of requests where the eviction filter dropped at "
    "least one token.",
)


def record_filter(raw_tokens: int, filtered_tokens: int) -> None:
    """Record one pass of the agent-tracker prompt-eviction filter.

    Args:
        raw_tokens: prompt_token_ids length seen at the eviction hook,
            before any tokens were dropped.
        filtered_tokens: number of tokens removed by the filter. May be 0
            if the hook ran but found nothing evictable.
    """
    if raw_tokens > 0:
        _AGENT_EVICTION_PREFILL_TOKENS_RAW.inc(raw_tokens)
    if filtered_tokens > 0:
        _AGENT_EVICTION_TOKENS_FILTERED.inc(filtered_tokens)
        _AGENT_EVICTION_FILTER_INVOCATIONS.inc()
