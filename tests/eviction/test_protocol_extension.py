"""Tests for the EvictionParams protocol extension.

Covers:
- PROTO-01: EvictionParams model defaults and validation
- PROTO-02: Backward compatibility — standard ChatCompletionRequest
             without eviction_params must continue to work unmodified
"""
import pytest
from pydantic import ValidationError

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    EvictionParams,
)


# ---------------------------------------------------------------------------
# PROTO-02 (most critical): standard requests must not break
# ---------------------------------------------------------------------------

def test_standard_request_without_eviction_params() -> None:
    """Standard ChatCompletionRequest with no eviction_params validates OK."""
    req = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hello"}],
    )
    assert req.eviction_params is None


# ---------------------------------------------------------------------------
# PROTO-01: EvictionParams defaults and field validation
# ---------------------------------------------------------------------------

def test_eviction_params_defaults() -> None:
    """EvictionParams() with no args creates instance with expected defaults."""
    params = EvictionParams()
    assert params.strategy is None
    assert params.keep_ratio == pytest.approx(0.7)
    assert params.eviction_interval_seconds == pytest.approx(3.0)
    assert params.eviction_delay_intervals == 0
    assert params.retention_window_tokens == 512
    assert params.prune_after_tokens == 512
    assert params.min_segment_tokens == 15
    assert params.protect_first_thought is True


def test_eviction_params_strategy_global() -> None:
    """EvictionParams(strategy='global') sets strategy correctly."""
    params = EvictionParams(strategy="global")
    assert params.strategy == "global"


def test_invalid_strategy_rejected() -> None:
    """EvictionParams with an unknown strategy raises ValidationError."""
    with pytest.raises(ValidationError):
        EvictionParams(strategy="invalid")


def test_keep_ratio_above_one_rejected() -> None:
    """EvictionParams(keep_ratio=1.5) raises ValidationError (> 1.0)."""
    with pytest.raises(ValidationError):
        EvictionParams(keep_ratio=1.5)


def test_keep_ratio_below_zero_rejected() -> None:
    """EvictionParams(keep_ratio=-0.1) raises ValidationError (< 0.0)."""
    with pytest.raises(ValidationError):
        EvictionParams(keep_ratio=-0.1)


def test_chat_completion_request_with_eviction_params() -> None:
    """ChatCompletionRequest with eviction_params dict validates and parses."""
    req = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hello"}],
        eviction_params={"strategy": "thought_min", "keep_ratio": 0.5},
    )
    assert req.eviction_params is not None
    assert req.eviction_params.strategy == "thought_min"
    assert req.eviction_params.keep_ratio == pytest.approx(0.5)


def test_all_four_strategies_accepted() -> None:
    """All four strategy literals are accepted by EvictionParams."""
    for strategy in ("global", "thought_min", "thought_avg", "random"):
        params = EvictionParams(strategy=strategy)
        assert params.strategy == strategy


# ---------------------------------------------------------------------------
# D-06: trigger_mode and eviction_interval_tokens extension
# ---------------------------------------------------------------------------

def test_trigger_mode_default_is_time() -> None:
    """EvictionParams() default trigger_mode is 'time'."""
    params = EvictionParams()
    assert params.trigger_mode == "time"


def test_eviction_interval_tokens_default() -> None:
    """EvictionParams() default eviction_interval_tokens is 256."""
    params = EvictionParams()
    assert params.eviction_interval_tokens == 256


def test_trigger_mode_token_round_trips() -> None:
    """EvictionParams(trigger_mode='token', eviction_interval_tokens=128) round-trips."""
    params = EvictionParams(trigger_mode="token", eviction_interval_tokens=128)
    assert params.trigger_mode == "token"
    assert params.eviction_interval_tokens == 128


def test_invalid_trigger_mode_rejected() -> None:
    """EvictionParams(trigger_mode='invalid') raises ValidationError."""
    with pytest.raises(ValidationError):
        EvictionParams(trigger_mode="invalid")


def test_eviction_interval_tokens_zero_rejected() -> None:
    """EvictionParams(eviction_interval_tokens=0) raises ValidationError (ge=1)."""
    with pytest.raises(ValidationError):
        EvictionParams(eviction_interval_tokens=0)


def test_existing_fields_preserve_defaults() -> None:
    """After adding trigger_mode, existing fields retain their original defaults."""
    params = EvictionParams()
    assert params.strategy is None
    assert params.keep_ratio == pytest.approx(0.7)
    assert params.eviction_interval_seconds == pytest.approx(3.0)
    assert params.eviction_delay_intervals == 0
    assert params.retention_window_tokens == 512
    assert params.prune_after_tokens == 512
    assert params.min_segment_tokens == 15
    assert params.protect_first_thought is True
