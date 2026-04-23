"""
Tests for vllm.thought_eviction.orchestrator.EvictionOrchestrator.

Coverage:
- Test 1:  _accumulate extends accumulated_l2_norms from res.new_l2_norms
- Test 2:  _accumulate extracts reasoning_content from <think>...</think>
- Test 3:  _accumulate computes reasoning_start_token_offset from prompt_token_ids
- Test 4:  _maybe_schedule_cycle respects time-based trigger
- Test 5:  _maybe_schedule_cycle respects token-count trigger
- Test 6:  _maybe_schedule_cycle skips when pending task is running
- Test 7:  _run_eviction_cycle returns early when prune_after_tokens not met (ENG-09)
- Test 8:  _run_eviction_cycle returns early when eviction_delay_intervals not met (ENG-10)
- Test 9:  _run_eviction_cycle returns early when generation_finished (Pitfall #6)
- Test 10: _run_eviction_cycle calls update_request_mask with absolute ranges (D-05)
- Test 11: _run_eviction_cycle accumulates permanently_evicted_ranges across calls (ENG-06)
- Test 12: wrap_stream yields all RequestOutput items unchanged (passthrough)
- Test 13: wrap_stream cancels pending task in finally block
- Test 14: Per-request isolation — two orchestrator instances share no state (ENG-07)
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_completion_output(text: str = "", token_ids: list | None = None):
    """Build a minimal CompletionOutput for testing."""
    from vllm.outputs import CompletionOutput
    return CompletionOutput(
        index=0,
        text=text,
        token_ids=token_ids or [],
        cumulative_logprob=None,
        logprobs=None,
        finish_reason=None,
    )


def make_request_output(
    text: str = "",
    new_l2_norms: list | None = None,
    prompt_token_ids: list | None = None,
    finished: bool = False,
    token_ids: list | None = None,
):
    """Build a minimal RequestOutput for testing.

    Args:
        text: Delta text for the completion output.
        new_l2_norms: Differential L2 norms attached to this output step.
        prompt_token_ids: Prompt token ids (None if not yet available).
        finished: Whether this output marks end of generation.
        token_ids: Token IDs in the completion output.
    """
    from vllm.outputs import RequestOutput
    completion = make_completion_output(text=text, token_ids=token_ids)
    return RequestOutput(
        request_id="req-test",
        prompt=None,
        prompt_token_ids=prompt_token_ids,
        prompt_logprobs=None,
        outputs=[completion],
        finished=finished,
        new_l2_norms=new_l2_norms,
    )


def make_eviction_params(**overrides):
    """Build EvictionParams with safe test defaults.

    Defaults: strategy=thought_min, small thresholds for fast testing.
    """
    from vllm.entrypoints.openai.chat_completion.protocol import EvictionParams
    defaults = dict(
        strategy="thought_min",
        keep_ratio=0.5,
        eviction_interval_seconds=0.05,   # 50ms for fast triggering in tests
        eviction_delay_intervals=0,
        retention_window_tokens=0,         # disable retention window by default
        prune_after_tokens=10,
        min_segment_tokens=5,
        protect_first_thought=True,
        trigger_mode="time",
        eviction_interval_tokens=20,
    )
    defaults.update(overrides)
    return EvictionParams(**defaults)


def make_mock_tokenizer(tokens_per_5_chars: bool = True):
    """Build a mock tokenizer returning ~1 token per 5 characters.

    Returns a callable that accepts (text, add_special_tokens, return_offsets_mapping)
    and returns a dict with input_ids and optionally offset_mapping.
    """
    tokenizer = MagicMock()

    def tokenizer_call(text, add_special_tokens=False, return_offsets_mapping=False):
        token_ids = []
        offsets = []
        pos = 0
        while pos < len(text):
            end = min(pos + 5, len(text))
            token_ids.append(pos)
            offsets.append((pos, end))
            pos = end
        result = {"input_ids": token_ids}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result

    tokenizer.side_effect = tokenizer_call
    return tokenizer


def make_orchestrator(
    params=None,
    engine_client=None,
    tokenizer=None,
    request_id: str = "req-test",
    block_size: int = 16,
):
    """Build an EvictionOrchestrator with test defaults."""
    from vllm.thought_eviction.orchestrator import EvictionOrchestrator
    if params is None:
        params = make_eviction_params()
    if engine_client is None:
        engine_client = AsyncMock()
        engine_client.update_request_mask = AsyncMock(return_value=True)
    if tokenizer is None:
        tokenizer = make_mock_tokenizer()
    return EvictionOrchestrator(
        eviction_params=params,
        engine_client=engine_client,
        tokenizer=tokenizer,
        request_id=request_id,
        block_size=block_size,
    )


# ---------------------------------------------------------------------------
# Test 1: _accumulate extends accumulated_l2_norms
# ---------------------------------------------------------------------------


def test_accumulate_extends_l2_norms():
    """Test 1: _accumulate extends accumulated_l2_norms from res.new_l2_norms."""
    orc = make_orchestrator()
    res = make_request_output(text="hello", new_l2_norms=[0.1, 0.2, 0.3])
    orc._accumulate(res)
    assert orc.accumulated_l2_norms == [0.1, 0.2, 0.3]


def test_accumulate_extends_l2_norms_multiple_steps():
    """Test 1b: multiple accumulate calls extend the list cumulatively."""
    orc = make_orchestrator()
    orc._accumulate(make_request_output(text="a", new_l2_norms=[0.1, 0.2]))
    orc._accumulate(make_request_output(text="b", new_l2_norms=[0.3, 0.4]))
    assert orc.accumulated_l2_norms == [0.1, 0.2, 0.3, 0.4]


def test_accumulate_noop_when_new_l2_norms_none():
    """Test 1c: new_l2_norms=None does not change accumulated list."""
    orc = make_orchestrator()
    orc._accumulate(make_request_output(text="hello", new_l2_norms=None))
    assert orc.accumulated_l2_norms == []


# ---------------------------------------------------------------------------
# Test 2: _accumulate extracts reasoning_content
# ---------------------------------------------------------------------------


def test_accumulate_extracts_reasoning_content():
    """Test 2: _accumulate extracts content between <think> and </think>."""
    orc = make_orchestrator()
    orc._accumulate(make_request_output(text="<think>step one</think>"))
    assert "step one" in orc.reasoning_content


def test_accumulate_extracts_reasoning_content_incremental():
    """Test 2b: reasoning_content accumulates across multiple delta steps."""
    orc = make_orchestrator()
    orc._accumulate(make_request_output(text="<think>part A"))
    orc._accumulate(make_request_output(text=" part B"))
    orc._accumulate(make_request_output(text="</think>"))
    assert "part A" in orc.reasoning_content
    assert "part B" in orc.reasoning_content


def test_accumulate_no_reasoning_content_without_think_tag():
    """Test 2c: Without <think> tag, reasoning_content stays empty."""
    orc = make_orchestrator()
    orc._accumulate(make_request_output(text="plain answer text"))
    assert orc.reasoning_content == ""


# ---------------------------------------------------------------------------
# Test 3: _accumulate computes reasoning_start_token_offset
# ---------------------------------------------------------------------------


def test_accumulate_computes_reasoning_start_offset():
    """Test 3: reasoning_start_token_offset computed from prompt_token_ids + prefix."""
    tokenizer = make_mock_tokenizer()
    orc = make_orchestrator(tokenizer=tokenizer)

    # Prompt has 5 tokens. Delta is "<think>reasoning" with "<think>" at position 0.
    # prefix_text = "<think>" (7 chars) → ceil(7/5) = 2 tokens in mock tokenizer
    prompt_ids = [10, 11, 12, 13, 14]  # 5 tokens
    orc._accumulate(make_request_output(
        text="<think>reasoning",
        prompt_token_ids=prompt_ids,
    ))

    assert orc.reasoning_start_token_offset is not None
    # prefix_text = "<think>" → 7 chars → 2 tokens (ceil at 5)
    # offset = 5 (prompt) + 2 (prefix tokens including <think>) = 7
    assert orc.reasoning_start_token_offset == 7


def test_accumulate_offset_computed_once():
    """Test 3b: reasoning_start_token_offset is not recomputed on subsequent calls."""
    tokenizer = make_mock_tokenizer()
    orc = make_orchestrator(tokenizer=tokenizer)

    prompt_ids = [10, 11, 12]
    orc._accumulate(make_request_output(text="<think>step", prompt_token_ids=prompt_ids))
    first_offset = orc.reasoning_start_token_offset

    orc._accumulate(make_request_output(text=" more text", prompt_token_ids=prompt_ids))
    assert orc.reasoning_start_token_offset == first_offset


# ---------------------------------------------------------------------------
# Test 4: _maybe_schedule_cycle — time-based trigger
# ---------------------------------------------------------------------------


def test_maybe_schedule_cycle_fires_after_time_interval():
    """Test 4: _maybe_schedule_cycle fires when time elapsed >= interval."""
    params = make_eviction_params(trigger_mode="time", eviction_interval_seconds=0.001)
    orc = make_orchestrator(params=params)
    # Force the last eviction time to be far in the past
    orc.last_eviction_time = time.monotonic() - 1.0

    fake_task = MagicMock()
    with patch("asyncio.create_task", return_value=fake_task) as mock_create:
        orc._maybe_schedule_cycle()
        mock_create.assert_called_once()
    assert orc._pending_task is fake_task


def test_maybe_schedule_cycle_does_not_fire_before_time():
    """Test 4b: _maybe_schedule_cycle does NOT fire before time interval."""
    params = make_eviction_params(trigger_mode="time", eviction_interval_seconds=1000.0)
    orc = make_orchestrator(params=params)

    with patch("asyncio.create_task") as mock_create:
        orc._maybe_schedule_cycle()
        mock_create.assert_not_called()
    assert orc._pending_task is None


# ---------------------------------------------------------------------------
# Test 5: _maybe_schedule_cycle — token-count trigger
# ---------------------------------------------------------------------------


def test_maybe_schedule_cycle_fires_after_token_count():
    """Test 5: _maybe_schedule_cycle fires when token count >= interval."""
    params = make_eviction_params(trigger_mode="token", eviction_interval_tokens=10)
    orc = make_orchestrator(params=params)
    orc.token_count_since_last_cycle = 15  # exceeds threshold

    fake_task = MagicMock()
    with patch("asyncio.create_task", return_value=fake_task) as mock_create:
        orc._maybe_schedule_cycle()
        mock_create.assert_called_once()
    assert orc._pending_task is fake_task


def test_maybe_schedule_cycle_does_not_fire_under_token_threshold():
    """Test 5b: _maybe_schedule_cycle skips when token count below threshold."""
    params = make_eviction_params(trigger_mode="token", eviction_interval_tokens=50)
    orc = make_orchestrator(params=params)
    orc.token_count_since_last_cycle = 5  # below threshold

    with patch("asyncio.create_task") as mock_create:
        orc._maybe_schedule_cycle()
        mock_create.assert_not_called()
    assert orc._pending_task is None


# ---------------------------------------------------------------------------
# Test 6: _maybe_schedule_cycle — skips when pending task running
# ---------------------------------------------------------------------------


def test_maybe_schedule_cycle_skips_if_pending_task_running():
    """Test 6: _maybe_schedule_cycle skips if a pending task is still running."""
    params = make_eviction_params(trigger_mode="time", eviction_interval_seconds=0.001)
    orc = make_orchestrator(params=params)
    orc.last_eviction_time = time.monotonic() - 1.0

    # Create a mock task that reports as not done
    fake_task = MagicMock()
    fake_task.done.return_value = False
    orc._pending_task = fake_task

    with patch("asyncio.create_task") as mock_create:
        orc._maybe_schedule_cycle()
        mock_create.assert_not_called()
    # _pending_task should remain the same (not replaced)
    assert orc._pending_task is fake_task


# ---------------------------------------------------------------------------
# Test 7: _run_eviction_cycle — ENG-09 guard (prune_after_tokens)
# ---------------------------------------------------------------------------


def test_run_eviction_cycle_early_return_prune_after_tokens():
    """Test 7: _run_eviction_cycle returns without calling engine when L2 norms below threshold."""
    engine_client = AsyncMock()
    engine_client.update_request_mask = AsyncMock(return_value=True)

    params = make_eviction_params(prune_after_tokens=100)
    orc = make_orchestrator(params=params, engine_client=engine_client)

    # Only 5 norms, threshold is 100
    orc.accumulated_l2_norms = [0.1] * 5
    orc.reasoning_content = "Some reasoning text"
    orc.reasoning_start_token_offset = 10

    asyncio.run(orc._run_eviction_cycle())
    engine_client.update_request_mask.assert_not_called()


# ---------------------------------------------------------------------------
# Test 8: _run_eviction_cycle — ENG-10 guard (eviction_delay_intervals)
# ---------------------------------------------------------------------------


def test_run_eviction_cycle_early_return_delay_intervals():
    """Test 8: _run_eviction_cycle returns early when cycle_count < eviction_delay_intervals."""
    engine_client = AsyncMock()
    engine_client.update_request_mask = AsyncMock(return_value=True)

    params = make_eviction_params(eviction_delay_intervals=3, prune_after_tokens=5)
    orc = make_orchestrator(params=params, engine_client=engine_client)
    orc.cycle_count = 0  # below delay threshold of 3
    orc.accumulated_l2_norms = [0.5] * 20
    orc.reasoning_content = "Some reasoning text here"
    orc.reasoning_start_token_offset = 10

    asyncio.run(orc._run_eviction_cycle())
    engine_client.update_request_mask.assert_not_called()
    # cycle_count should have been incremented
    assert orc.cycle_count == 1


# ---------------------------------------------------------------------------
# Test 9: _run_eviction_cycle — Pitfall #6 guard (generation_finished)
# ---------------------------------------------------------------------------


def test_run_eviction_cycle_early_return_when_generation_finished():
    """Test 9: _run_eviction_cycle returns without evicting when generation_finished."""
    engine_client = AsyncMock()
    engine_client.update_request_mask = AsyncMock(return_value=True)

    params = make_eviction_params(prune_after_tokens=5)
    orc = make_orchestrator(params=params, engine_client=engine_client)
    orc._generation_finished = True
    orc.accumulated_l2_norms = [0.5] * 50
    orc.reasoning_content = "Some lengthy reasoning content to segment"
    orc.reasoning_start_token_offset = 10

    asyncio.run(orc._run_eviction_cycle())
    engine_client.update_request_mask.assert_not_called()


# ---------------------------------------------------------------------------
# Test 10: _run_eviction_cycle — absolute offset applied (D-05)
# ---------------------------------------------------------------------------


def test_run_eviction_cycle_applies_absolute_offset():
    """Test 10: _run_eviction_cycle calls update_request_mask with absolute ranges."""
    engine_client = AsyncMock()
    engine_client.update_request_mask = AsyncMock(return_value=True)

    # Use global strategy to simplify range prediction
    params = make_eviction_params(
        strategy="global",
        keep_ratio=0.0,         # evict everything
        prune_after_tokens=5,
        retention_window_tokens=0,
    )
    tokenizer = make_mock_tokenizer()
    orc = make_orchestrator(params=params, engine_client=engine_client, tokenizer=tokenizer)

    # Set up state: 37 tokens worth of norms, offset=5
    # norms[5:] = 32 norms available to the strategy
    orc.accumulated_l2_norms = [float(i) for i in range(37)]  # 37 norms
    orc.reasoning_content = "Reasoning " * 3
    orc.reasoning_start_token_offset = 5  # absolute offset
    orc._generation_finished = False

    asyncio.run(orc._run_eviction_cycle())

    # update_request_mask must have been called
    engine_client.update_request_mask.assert_called_once()
    call_args = engine_client.update_request_mask.call_args
    request_id_arg = call_args[0][0]
    absolute_ranges = call_args[0][1]

    assert request_id_arg == "req-test"
    # All ranges must be offset by 5
    for start, end in absolute_ranges:
        assert start >= 5, f"Range start {start} is not offset by 5"
        assert end > start


# ---------------------------------------------------------------------------
# Test 11: _run_eviction_cycle — permanently_evicted_ranges accumulates (ENG-06)
# ---------------------------------------------------------------------------


def test_run_eviction_cycle_accumulates_permanently_evicted_ranges():
    """Test 11: permanently_evicted_ranges accumulates merged ranges across cycles."""
    engine_client = AsyncMock()
    engine_client.update_request_mask = AsyncMock(return_value=True)

    params = make_eviction_params(
        strategy="global",
        keep_ratio=0.0,
        prune_after_tokens=5,
        retention_window_tokens=0,
    )
    tokenizer = make_mock_tokenizer()
    orc = make_orchestrator(params=params, engine_client=engine_client, tokenizer=tokenizer)

    orc.accumulated_l2_norms = [float(i) for i in range(32)]
    orc.reasoning_content = "Reasoning " * 3
    orc.reasoning_start_token_offset = 50
    orc._generation_finished = False

    # First cycle
    asyncio.run(orc._run_eviction_cycle())
    ranges_after_first = list(orc.permanently_evicted_ranges)

    # Second cycle with more norms
    orc.accumulated_l2_norms = [float(i) for i in range(64)]
    asyncio.run(orc._run_eviction_cycle())

    # permanently_evicted_ranges should have grown or remained the same merged form
    assert len(orc.permanently_evicted_ranges) >= len(ranges_after_first)


# ---------------------------------------------------------------------------
# Test 12: wrap_stream — yields all items unchanged (passthrough)
# ---------------------------------------------------------------------------


def test_wrap_stream_yields_all_items_unchanged():
    """Test 12: wrap_stream is a passthrough — yields each RequestOutput unchanged."""
    orc = make_orchestrator(params=make_eviction_params(
        trigger_mode="time",
        eviction_interval_seconds=1000.0,  # prevent any triggers
    ))

    outputs = [
        make_request_output(text="hello", new_l2_norms=[0.1]),
        make_request_output(text=" world", new_l2_norms=[0.2]),
        make_request_output(text=".", new_l2_norms=[0.3], finished=True),
    ]

    async def mock_generator():
        for o in outputs:
            yield o

    async def run():
        collected = []
        async for res in orc.wrap_stream(mock_generator()):
            collected.append(res)
        return collected

    collected = asyncio.run(run())

    assert len(collected) == 3
    assert collected[0] is outputs[0]
    assert collected[1] is outputs[1]
    assert collected[2] is outputs[2]


# ---------------------------------------------------------------------------
# Test 13: wrap_stream — cancels pending task in finally block
# ---------------------------------------------------------------------------


def test_wrap_stream_cancels_pending_task_on_exit():
    """Test 13: wrap_stream cancels pending eviction task in finally block.

    Verifies that wrap_stream's finally block calls cancel() on any pending
    eviction cycle task when the stream generator is exhausted.
    """
    orc = make_orchestrator(params=make_eviction_params(
        trigger_mode="time",
        eviction_interval_seconds=1000.0,  # prevent auto-triggering
    ))

    cancel_called = {"value": False}

    async def fake_awaitable():
        raise asyncio.CancelledError()

    class FakeTask:
        """Minimal Task-like object that tracks cancel() calls."""
        def done(self):
            return False

        def cancel(self, msg=None):
            cancel_called["value"] = True

        def __await__(self):
            return fake_awaitable().__await__()

    # Pre-set a pending task to simulate a running cycle
    orc._pending_task = FakeTask()

    async def mock_generator():
        yield make_request_output(text="hello", new_l2_norms=[0.1])

    async def run():
        async for _ in orc.wrap_stream(mock_generator()):
            pass

    asyncio.run(run())
    # cancel() must have been called in the finally block
    assert cancel_called["value"] is True


# ---------------------------------------------------------------------------
# Test 14: Per-request isolation (ENG-07)
# ---------------------------------------------------------------------------


def test_per_request_isolation():
    """Test 14: Two EvictionOrchestrator instances share no state."""
    orc1 = make_orchestrator(request_id="req-1")
    orc2 = make_orchestrator(request_id="req-2")

    orc1._accumulate(make_request_output(text="<think>thoughts A", new_l2_norms=[1.0, 2.0]))
    orc1._accumulate(make_request_output(text=" more A", new_l2_norms=[3.0]))

    # orc2 should not have any norms from orc1
    assert orc2.accumulated_l2_norms == []
    assert orc2.reasoning_content == ""
    assert orc2.accumulated_text == ""

    # orc1 should have its own norms
    assert orc1.accumulated_l2_norms == [1.0, 2.0, 3.0]
    assert "thoughts A" in orc1.reasoning_content

    # Modifying orc2 should not affect orc1
    orc2._accumulate(make_request_output(text="completely different", new_l2_norms=[9.9]))
    assert orc1.accumulated_l2_norms == [1.0, 2.0, 3.0]
    assert "completely different" not in orc1.reasoning_content
