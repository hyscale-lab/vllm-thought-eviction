"""
EvictionOrchestrator: async middleware for server-side KV cache eviction.

This module implements the core of Phase 2: a per-request orchestrator that
wraps the streaming RequestOutput generator, accumulates L2 norms and reasoning
content incrementally, and fires non-blocking eviction cycles as background
asyncio tasks.

Key design decisions (per design doc):
- D-01: wrap_stream intercepts each RequestOutput without blocking the generator
- D-02: One EvictionOrchestrator instance per request — all state is isolated
- D-04: Eviction commands go directly through engine_client.update_request_mask
- D-05: Reasoning-relative token positions are converted to absolute before engine call
- D-03: L2 norms are accumulated differentially from RequestOutput.new_l2_norms
"""

import asyncio
import regex as re
import time
from typing import AsyncIterator, Optional

import numpy as np

from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.thought_eviction.segmenter import ThoughtSegmenter
from vllm.thought_eviction.block_utils import (
    merge_overlapping_ranges,
    align_ranges_to_blocks,
    apply_retention_window,
)
from vllm.thought_eviction.strategies import (
    GlobalStrategy,
    ThoughtMinStrategy,
    ThoughtAvgStrategy,
    RandomStrategy,
)
from vllm.entrypoints.openai.chat_completion.protocol import EvictionParams
from vllm.v1.attention.l2_norm_cache import get_l2_norm_cache

logger = init_logger(__name__)

_THINK_START_RE = re.compile(r'<think>', re.IGNORECASE)
_THINK_END_RE = re.compile(r'</think>', re.IGNORECASE)


def _build_strategy(params: EvictionParams):
    """Instantiate the eviction strategy specified in EvictionParams.

    Args:
        params: EvictionParams from the chat completion request.

    Returns:
        Strategy instance for the requested mode.

    Raises:
        ValueError: If params.strategy is not a recognized value.
    """
    strategy_name = params.strategy
    if strategy_name == 'global':
        return GlobalStrategy()
    elif strategy_name == 'thought_min':
        return ThoughtMinStrategy()
    elif strategy_name == 'thought_avg':
        return ThoughtAvgStrategy()
    elif strategy_name == 'random':
        return RandomStrategy()
    else:
        raise ValueError(f"Unknown eviction strategy: {strategy_name!r}")


class EvictionOrchestrator:
    """Per-request middleware that wraps the result generator and runs eviction.

    One instance is created per eviction-enabled request and discarded when
    the request completes. All mutable state is strictly per-instance — there
    is no shared state between requests.

    Eviction cycles are scheduled as asyncio background tasks (non-blocking).
    The stream generator is never paused or interrupted by an eviction cycle.

    Args:
        eviction_params: EvictionParams from the chat completion request.
        engine_client: AsyncLLM instance providing update_request_mask().
        tokenizer: Pre-loaded tokenizer instance (vLLM TokenizerGroup or
            compatible callable supporting return_offsets_mapping=True).
        request_id: Unique request identifier for mask updates.
        block_size: KV cache block size in tokens for block-aligned eviction.
    """

    def __init__(
        self,
        eviction_params: EvictionParams,
        engine_client,
        tokenizer,
        request_id: str,
        block_size: int,
    ) -> None:
        self.params = eviction_params
        self.engine_client = engine_client
        self.tokenizer = tokenizer
        self.request_id = request_id
        self.block_size = block_size

        # Phase 1 components
        self.segmenter = ThoughtSegmenter(tokenizer, eviction_params.min_segment_tokens)
        self.strategy = _build_strategy(eviction_params)

        # Per-request mutable state (D-02: all isolated to this instance)
        self.accumulated_l2_norms: list[float] = []
        self.permanently_evicted_ranges: list[tuple[int, int]] = []
        self.cycle_count: int = 0
        self.last_eviction_time: float = time.monotonic()
        self.token_count_since_last_cycle: int = 0
        self.reasoning_start_token_offset: Optional[int] = None
        self.accumulated_text: str = ""
        self.reasoning_content: str = ""
        self._in_think_block: bool = False
        self._think_start_found: bool = False
        self._generation_finished: bool = False
        self._pending_task: Optional[asyncio.Task] = None

        # Phase 4: per-cycle eviction event accumulation for stats payload
        self._eviction_events: list[dict] = []
        self._start_time: float = time.monotonic()

    async def wrap_stream(
        self,
        result_generator: AsyncIterator[RequestOutput],
    ) -> AsyncIterator[RequestOutput]:
        """Wrap the result generator, accumulating state and scheduling cycles.

        Yields each RequestOutput unchanged (middleware passthrough). Eviction
        cycles are fired as background tasks and do not delay yield.

        Per D-01: intercept each item, update state, maybe schedule a cycle,
        then yield. The finally block cancels any pending cycle on early exit
        (e.g., client disconnect, exception).

        Args:
            result_generator: Async generator of RequestOutput from the engine.

        Yields:
            Each RequestOutput item from result_generator, unchanged.
        """
        try:
            async for res in result_generator:
                self._accumulate(res)
                self._maybe_schedule_cycle()
                if res.finished:
                    self._generation_finished = True
                yield res
        finally:
            if self._pending_task and not self._pending_task.done():
                self._pending_task.cancel()
                try:
                    await self._pending_task
                except asyncio.CancelledError:
                    pass
            # D-05: clean up per-request L2NormCache data (both _request_data and _request_layer_prefs)
            get_l2_norm_cache().remove_request(self.request_id)

    def _accumulate(self, res: RequestOutput) -> None:
        """Accumulate text, L2 norms, and reasoning content from one output.

        Updates accumulated_text, reasoning_content, accumulated_l2_norms,
        token_count_since_last_cycle, and reasoning_start_token_offset.

        Args:
            res: A RequestOutput from the engine for this request.
        """
        # Extract delta text from outputs
        delta_text = ""
        if res.outputs:
            delta_text = res.outputs[0].text or ""

        if delta_text:
            self.accumulated_text += delta_text

            # Track <think> tag entry (only once per request)
            if not self._think_start_found:
                start_match = _THINK_START_RE.search(self.accumulated_text)
                if start_match:
                    self._think_start_found = True
                    self._in_think_block = True

            # Compute reasoning_start_token_offset once prompt token ids are available.
            self._maybe_set_reasoning_start_token_offset(res)

            # Detect </think> tag to stop tracking reasoning content
            if self._in_think_block:
                end_match = _THINK_END_RE.search(self.accumulated_text)
                if end_match:
                    self._in_think_block = False

            # Update reasoning_content from full accumulated text
            if self._think_start_found:
                start_match = _THINK_START_RE.search(self.accumulated_text)
                if start_match:
                    end_match = _THINK_END_RE.search(self.accumulated_text, start_match.end())
                    if end_match:
                        self.reasoning_content = self.accumulated_text[
                            start_match.end():end_match.start()
                        ]
                    else:
                        self.reasoning_content = self.accumulated_text[start_match.end():]

        # Count new tokens for the token-based trigger.
        if res.outputs:
            token_ids = getattr(res.outputs[0], 'token_ids', None)
            if token_ids is not None:
                # token_ids is the per-step token delta for this stream update.
                self.token_count_since_last_cycle += len(token_ids)
            elif delta_text:
                # Fallback when token ids are unavailable: approximate from text size.
                self.token_count_since_last_cycle += max(1, len(delta_text) // 4)

        # Extend L2 norms differentially (D-03, ENG-01)
        if res.new_l2_norms:
            self.accumulated_l2_norms.extend(res.new_l2_norms)

    def _maybe_set_reasoning_start_token_offset(self, res: RequestOutput) -> None:
        """Set reasoning_start_token_offset once <think> and prompt tokens are known."""
        if not self._think_start_found:
            return
        if self.reasoning_start_token_offset is not None:
            return
        if res.prompt_token_ids is None:
            return

        start_match = _THINK_START_RE.search(self.accumulated_text)
        if not start_match:
            return

        prompt_len = len(res.prompt_token_ids)
        prefix_text = self.accumulated_text[:start_match.end()]
        encoding = self.tokenizer(
            prefix_text,
            add_special_tokens=False,
            return_offsets_mapping=False,
        )
        prefix_token_count = len(encoding['input_ids'])

        self.reasoning_start_token_offset = prompt_len + prefix_token_count

    def _maybe_schedule_cycle(self) -> None:
        """Conditionally schedule an eviction cycle as a background task.

        Skips if a cycle task is still running. Checks the configured
        trigger_mode ('time' or 'token') to decide whether to fire.
        Resets the relevant counter after scheduling.
        """
        # Don't overlap cycles
        if self._pending_task and not self._pending_task.done():
            return

        should_trigger = False
        if self.params.trigger_mode == 'time':
            elapsed = time.monotonic() - self.last_eviction_time
            should_trigger = elapsed >= self.params.eviction_interval_seconds
        else:  # 'token'
            should_trigger = (
                self.token_count_since_last_cycle >= self.params.eviction_interval_tokens
            )

        if should_trigger:
            self.last_eviction_time = time.monotonic()
            self.token_count_since_last_cycle = 0
            self._pending_task = asyncio.create_task(self._run_eviction_cycle())

    async def _run_eviction_cycle(self) -> None:
        """Run one eviction cycle as a background task.

        Segments reasoning content, assigns L2 norms to thoughts, computes
        evictable ranges via the strategy, applies retention/block alignment,
        converts to absolute token positions, and sends to the engine.

        Guard conditions (in order):
        1. ENG-09: Not enough L2 norms accumulated yet
        2. ENG-10: Cycle delayed by eviction_delay_intervals
        3. No reasoning content yet
        4. Generation already finished (Pitfall #6)
        5. reasoning_start_token_offset not yet computed

        All errors are caught and logged to avoid crashing the stream.
        """
        try:
            # Guard 1 (ENG-09): minimum token threshold
            # Random strategy selects thoughts without L2 norms — skip norm-count guard.
            if self.params.strategy != "random":
                if len(self.accumulated_l2_norms) < self.params.prune_after_tokens:
                    return

            # Guard 2 (ENG-10): delay intervals
            if self.cycle_count < self.params.eviction_delay_intervals:
                self.cycle_count += 1
                return

            # Guard 3: need reasoning content
            if not self.reasoning_content:
                return

            # Guard 4 (Pitfall #6): don't evict after generation finished
            if self._generation_finished:
                return

            # Guard 5: offset must be known to compute absolute positions
            if self.reasoning_start_token_offset is None:
                return

            # Segment reasoning content into thoughts
            thoughts = self.segmenter.update(self.reasoning_content)

            # Assign L2 norms to thoughts (ENG-02)
            l2_array = np.array(self.accumulated_l2_norms[self.reasoning_start_token_offset:], dtype=np.float32)
            for thought in thoughts:
                start = thought.start_token_pos
                end = thought.end_token_pos
                if start >= 0 and end > start and start < len(l2_array) and end <= len(l2_array):
                    thought_norms = l2_array[start:end]
                    thought.l2_norms = thought_norms
                    thought.min_l2_norm = float(np.min(thought_norms))
                    thought.avg_l2_norm = float(np.mean(thought_norms))

            # Compute evictable ranges via strategy
            strategy_name = self.params.strategy
            ranges: list[tuple[int, int]] = []
            if strategy_name == 'global':
                ranges = self.strategy.compute_evictable_ranges(
                    l2_array,
                    self.params.keep_ratio,
                )
            else:
                # Thought-based strategies (thought_min, thought_avg, random)
                ranges = self.strategy.compute_evictable_ranges(
                    thoughts,
                    self.params.keep_ratio,
                    self.params.prune_after_tokens,
                    self.params.protect_first_thought,
                )

            if not ranges:
                self.cycle_count += 1
                return

            # Merge with permanently evicted ranges (ENG-06)
            all_ranges = self.permanently_evicted_ranges + ranges
            merged = merge_overlapping_ranges(all_ranges)

            # Apply retention window (ENG-05): protect the last N reasoning tokens
            # Only apply when retention_window_tokens > 0 AND enough tokens exist.
            # When retention_floor would be 0, all ranges would be discarded — skip.
            total_reasoning_tokens = len(l2_array)
            retention_window = self.params.retention_window_tokens
            if retention_window > 0 and total_reasoning_tokens > retention_window:
                retention_floor = total_reasoning_tokens - retention_window
                protected = apply_retention_window(merged, retention_floor)
            else:
                protected = merged

            # Align to KV cache block boundaries (ENG-04)
            aligned = align_ranges_to_blocks(protected, self.block_size)

            if not aligned:
                self.permanently_evicted_ranges = merged
                self.cycle_count += 1
                return

            # Convert reasoning-relative to absolute token positions (D-05)
            offset = self.reasoning_start_token_offset
            absolute = [(s + offset, e + offset) for s, e in aligned]

            # Send eviction command to engine (D-04)
            await self.engine_client.update_request_mask(self.request_id, absolute)

            # Update permanent eviction record
            self.permanently_evicted_ranges = merged

            self.cycle_count += 1

            evicted_tokens = sum(e - s for s, e in aligned)

            # Phase 4: accumulate per-cycle eviction event for stats payload
            self._eviction_events.append({
                "interval_number": self.cycle_count,
                "tokens_evicted": evicted_tokens,
                "ranges": [[s, e] for s, e in aligned],
                "timestamp": round(time.monotonic() - self._start_time, 4),
            })

            logger.info(
                "Eviction cycle %d for %s: %d ranges, %d tokens evicted",
                self.cycle_count,
                self.request_id,
                len(aligned),
                evicted_tokens,
            )

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(
                "Eviction cycle failed for request %s: %s",
                self.request_id,
                exc,
                exc_info=True,
            )

    def build_eviction_payload(self) -> dict:
        """Build eviction statistics payload for the final SSE chunk.

        Computes summary stats from the final segmenter state and returns
        the full eviction payload matching the L2NormEvictionProcessor schema.

        Returns:
            Dict with 'summary', 'events', and 'masked_tokens' keys.
            Returns zero-count summary if no eviction cycles completed.
        """
        masked_tokens = sum(e - s for s, e in self.permanently_evicted_ranges)

        # Compute L2 summary from final segmenter state
        thoughts = []
        if self.reasoning_content:
            try:
                thoughts = self.segmenter.update(self.reasoning_content)
            except Exception:
                pass

        evicted = [t for t in thoughts if t.evicted and t.min_l2_norm is not None]
        kept = [t for t in thoughts if not t.evicted and t.min_l2_norm is not None]

        summary = {
            "total_thoughts": len(thoughts),
            "evicted_thoughts": len(evicted),
            "kept_thoughts": len(kept),
            "avg_min_l2_evicted": (
                round(float(np.mean([t.min_l2_norm for t in evicted])), 6)
                if evicted else None
            ),
            "avg_min_l2_kept": (
                round(float(np.mean([t.min_l2_norm for t in kept])), 6)
                if kept else None
            ),
            "avg_avg_l2_evicted": (
                round(float(np.mean([t.avg_l2_norm for t in evicted])), 6)
                if evicted else None
            ),
            "avg_avg_l2_kept": (
                round(float(np.mean([t.avg_l2_norm for t in kept])), 6)
                if kept else None
            ),
        }

        return {
            "summary": summary,
            "events": self._eviction_events,
            "masked_tokens": masked_tokens,
        }
