"""Live trajectory tracker (Phase 01.4) -- per-session SessionTracker + module
registry. Sync inline, no async, no engine_client coupling.

Algorithm: incremental port of scripts/trajectory_classifier.py:build_turn_states
+ classify_conversation. The driver loop is now O(new_msgs) per request with
bounded back-walk (D-08); the per-turn classification logic is unchanged from
the offline classifier.

NO-EVICT ZONE (D-10): the tracker computes the no-evict zone DYNAMICALLY at
session start (sum of leading system + first user message token-range lengths)
and stores it on the SessionTracker instance. The legacy 1,991-token p95
constant from Phase 01.1 is INTENTIONALLY NOT imported -- it lives only in the
offline eviction_analysis.py for legacy curve work. Naming the legacy
identifier here would defeat the literal-grep gate that guards D-10.
"""
from __future__ import annotations

import time
from collections import OrderedDict
from typing import Any

from vllm.logger import init_logger

from vllm.agent_tracker.classifier import (
    K_DEFAULTS,
    MSA_AGENT_REASONING,
    MSA_AGENT_TOOL_CALL,
    MSA_TOOL_FILE_READ,
    MSA_TOOL_FILE_EDIT,
    MSA_TOOL_FILE_SEARCH,
    MSA_TOOL_TEST_RUN,
    MSA_TOOL_BUILD_INSTALL,
    MSA_TOOL_RUN_EXEC,
    MSA_TOOL_OTHER,
    MSA_SYSTEM_PROMPT,
    MSA_USER_TASK,
    _classify_new_messages,
    _extract_files_from_messages,
    _files_overlap,
    _is_path_suffix,
    extract_new_messages,
    build_tool_call_command_map,
    classify_bash_command_minisweagent,
)
from vllm.agent_tracker.file_timeline import FileTimeline
from vllm.agent_tracker.segment_map import EvictableSegmentMap, TurnState
from vllm.agent_tracker.token_index import TokenSequenceIndex, hash_token_sequence

logger = init_logger(__name__)


# =============================================================================
# SECTION B -- compute_message_token_ranges helper (RESEARCH Finding 7)
# =============================================================================

# Chat-template compatibility shims. Mirrors
# scripts/analyze_tokens.py:_normalize_messages_for_chat_template, which solved
# the same two issues during offline prefix-overlap analysis:
#   (1) tool_call.arguments arrives as a JSON string from litellm, but the
#       Qwen jinja template iterates it with `.items()` and raises
#       `TypeError: Can only get item pairs from a mapping.` unless we coerce
#       to a dict first.
#   (2) When apply_chat_template is fed a sub-list with no non-tool-response
#       user message, the template raises `No user query found in messages.`
#       at the `last_query_index` check (chat_template.jinja:67-80). Inject a
#       placeholder user message just after any leading system block so the
#       check succeeds; the placeholder's contribution is subtracted out
#       afterwards to keep cumulative ranges aligned with the natural full
#       render.

_PLACEHOLDER_USER_CONTENT = "[placeholder for incremental rendering]"
_PLACEHOLDER_USER_MSG = {"role": "user", "content": _PLACEHOLDER_USER_CONTENT}


def _materialize_message(msg: Any) -> dict:
    """Return a plain dict copy of one message, eagerly materializing any
    pydantic v2 lazy-validator fields (notably `tool_calls: Iterable[...]`
    -- chat_utils.py:293). The Qwen jinja template (and all downstream
    classifier helpers) only need dict-shaped messages; we deliberately
    avoid `copy.deepcopy(msg)` because pydantic v2 validates `Iterable[...]`
    lazily into a `pydantic_core._pydantic_core.ValidatorIterator` which is
    NOT picklable (rust-backed) -- deepcopy walks fields via __reduce_ex__ /
    pickle and raises
    `TypeError: cannot pickle 'pydantic_core...ValidatorIterator'`.

    Caveat: a ValidatorIterator can only be consumed ONCE. If the caller
    held a reference to msg["tool_calls"] before materialization and tries
    to iterate it later, they'll get an exhausted iterator. The
    serving.py hook site MUST therefore materialize once and pass the
    materialized list to ALL downstream consumers (compute_message_token_ranges
    AND SessionTrackerRegistry.observe_request) -- not pass request.messages
    directly anywhere after this call.

    Strategy:
      1. If msg has .model_dump(), use it -- pydantic materializes ALL
         lazy iterators in the process.
      2. Otherwise (plain dict / TypedDict from tests), copy keys manually
         and force-list any iterable tool_calls field.
    """
    # Pydantic v2 model_dump materializes lazy validators. Use default mode
    # ('python') so nested pydantic objects also become plain dicts.
    if hasattr(msg, "model_dump") and callable(msg.model_dump):
        try:
            return msg.model_dump()
        except Exception:
            pass  # fall through to dict path

    if not isinstance(msg, dict):
        # Custom TypedDict-like object; coerce to dict.
        try:
            return dict(msg)
        except Exception:
            return {}

    out: dict = {}
    for k, v in msg.items():
        if k == "tool_calls" and v is not None:
            # Materialize lazy iterator into a list of plain dicts.
            materialized: list = []
            for tc in v:
                if hasattr(tc, "model_dump") and callable(tc.model_dump):
                    materialized.append(tc.model_dump())
                elif isinstance(tc, dict):
                    materialized.append(dict(tc))
                else:
                    try:
                        materialized.append(dict(tc))
                    except Exception:
                        materialized.append(tc)
            out[k] = materialized
        else:
            out[k] = v
    return out


def materialize_messages(messages: list) -> list[dict]:
    """Public helper: return a list of plain-dict copies of `messages`,
    eagerly materializing all pydantic v2 lazy iterators. Call once at the
    hook site and pass the result to ALL downstream tracker functions to
    avoid double-consumption of `tool_calls` ValidatorIterators (debug
    session agent-tracker-observe-no-user-query, Issue 2).
    """
    return [_materialize_message(m) for m in messages]


def _coerce_tool_call_arguments(messages: list) -> list[dict]:
    """Return a list of plain-dict copies of `messages` with every
    tool_call's `arguments` field coerced from JSON-encoded string to dict
    (mirrors offline helper section 1). Non-mapping JSON values are wrapped
    under a single-key dict so the template's `.items()` call still
    succeeds. Unparseable strings are left as-is so genuinely broken
    payloads still surface as exceptions.

    Pickling-safe: uses _materialize_message to eagerly materialize any
    pydantic v2 lazy `Iterable[...]` validators (debug session
    agent-tracker-observe-no-user-query, Issue 2). Idempotent if called on
    an already-materialized list.
    """
    import json as _json

    out: list[dict] = [_materialize_message(m) for m in messages]
    for msg in out:
        if not isinstance(msg, dict):
            continue
        tool_calls = msg.get("tool_calls")
        if not tool_calls or not isinstance(tool_calls, list):
            continue
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            target = (
                tc.get("function")
                if isinstance(tc.get("function"), dict)
                else tc
            )
            args = target.get("arguments")
            if isinstance(args, str):
                try:
                    parsed = _json.loads(args)
                    if isinstance(parsed, dict):
                        target["arguments"] = parsed
                    else:
                        target["arguments"] = {"_raw": parsed}
                except (_json.JSONDecodeError, TypeError):
                    pass
    return out


def _prefix_has_user(prefix: list[dict]) -> bool:
    """True if the prefix contains at least one role=user message whose content
    is not a <tool_response>...</tool_response> wrapper. Mirrors the Qwen jinja
    template's `last_query_index` reverse-walk (chat_template.jinja:67-80)."""
    for m in prefix:
        if not isinstance(m, dict) or m.get("role") != "user":
            continue
        content = m.get("content")
        if isinstance(content, str):
            stripped = content.strip()
            if stripped.startswith("<tool_response>") and stripped.endswith(
                "</tool_response>"
            ):
                continue
        return True
    return False


def _inject_placeholder_after_leading_system(prefix: list[dict]) -> list[dict]:
    """Return a copy of `prefix` with a placeholder user message inserted just
    after any leading role=system messages. Idempotent: if the prefix already
    contains a user message, returns it unchanged.

    Mirrors scripts/analyze_tokens.py:_normalize_messages_for_chat_template
    section (2). The placeholder satisfies the chat template's user-query check
    without disturbing the relative order of any real content.
    """
    if _prefix_has_user(prefix):
        return list(prefix)
    insert_at = 0
    for i, m in enumerate(prefix):
        if isinstance(m, dict) and m.get("role") == "system":
            insert_at = i + 1
        else:
            break
    out = list(prefix)
    out.insert(insert_at, dict(_PLACEHOLDER_USER_MSG))
    return out


def compute_message_token_ranges(
    messages: list,
    tokenizer,
    request,
    chat_template: str | None,
    chat_template_content_format: str,
    default_chat_template_kwargs: dict,
    tools: list[dict] | None = None,
    add_generation_prompt: bool = True,
    continue_final_message: bool = False,
) -> list[tuple[int, int]]:
    """Per-message token ranges via incremental re-tokenization (RESEARCH Finding 7).

    HIGHEST-RISK helper of Phase 01.4: apply_chat_template does NOT support
    return_offsets_mapping (only the raw `encode` path on fast tokenizers does).
    The cheapest reliable approach is to re-run the chat template on growing
    prefixes and record the running length. Each prefix tokenization includes
    the role-marker / separator tokens at message boundaries, so the resulting
    ranges sum exactly to the renderer's flat prompt length.

    Cost: ~50-100ms for 300-turn sessions. Acceptable for v1 (CONTEXT D-03 says
    smoke test must report tracker latency; revisit only if >50ms regression).

    The hook (Plan 04) MUST assert ``ranges[-1][1] == len(prompt_token_ids)``
    and gracefully degrade (logger.warning + skip the tracker for that request)
    when they disagree. This function is the helper; the assertion lives at the
    callsite where prompt_token_ids is known.

    NORMALIZATION (debug session: agent-tracker-observe-no-user-query):
    Two Qwen jinja chat-template quirks force pre-processing here.

    (a) tool_call.arguments arrives as a JSON-encoded string from litellm,
        which trips `TypeError: Can only get item pairs from a mapping.`
        inside the template. We coerce them to dict ONCE on the full message
        list before iterating prefixes; the coerced list is what we slice in
        the loop. The coercion is purely structural (string -> dict) and does
        not change the rendered token count -- tojson on a dict produces the
        same JSON string as the original.

    (b) When a prefix has no non-tool-response user message, the template
        raises 'No user query found in messages.' at the `last_query_index`
        check (chat_template.jinja:67-80). The very first iteration (i=1,
        messages[:1] = [system]) always trips this for mini-swe-agent. We
        inject a placeholder user message after any leading system block when
        a prefix lacks a user, render the augmented prefix, and subtract a
        precomputed placeholder overhead so the running length matches what
        the prefix WOULD render to without the placeholder. The final
        iteration (full messages list) has a real user, is rendered natively,
        and therefore satisfies the engine-side assertion
        `ranges[-1][1] == len(prompt_token_ids)` exactly.

    TOOLS (debug session Issue 1, 2026-04-28): the engine renders
    prompt_token_ids via safe_apply_chat_template(tools=tool_dicts, ...) where
    tool_dicts = [t.model_dump() for t in request.tools]. The Qwen jinja
    template's tools-preamble contributes a fixed per-render overhead inside
    the system block when tools is not None. This helper therefore MUST
    receive the same `tools` list and pass it on every apply_chat_template
    call -- including the placeholder-overhead probe (the probe is a
    standalone render, but the tools preamble is emitted unconditionally
    when tools is not none, so the probe length already includes the
    tools-preamble overhead and naturally cancels out via subtraction).

    ADD_GENERATION_PROMPT (debug session Round 3, 2026-04-28): the engine
    renders prompt_token_ids with request.add_generation_prompt (default
    True), which appends `<|im_start|>assistant\\n<think>\\n` (5 tokens on
    Qwen3.6 with default enable_thinking) to the end. Tracker MUST mirror
    this on the FINAL iteration so ranges[-1][1] absorbs the gen_prompt
    overhead and equals len(prompt_token_ids) exactly. Earlier iterations
    stay add_generation_prompt=False so the per-prefix increments measure
    only the new message's contribution; the gen_prompt tokens are
    attributed entirely to the last range (a tiny scaffolding bias that
    keeps the cumulative sum exact).
    """
    # NORMALIZATION (a): coerce tool_call.arguments once on the full list.
    # Also materializes pydantic v2 lazy iterators so subsequent iteration
    # is pickling-safe (Issue 2).
    norm_messages = _coerce_tool_call_arguments(messages)

    ranges: list[tuple[int, int]] = []
    prev_len = 0
    kwargs = {
        **default_chat_template_kwargs,
        **(getattr(request, "chat_template_kwargs", None) or {}),
    }
    # Strip add_generation_prompt / continue_final_message from kwargs if
    # present -- we set them per-iteration ourselves (see Round 3 docstring).
    kwargs.pop("add_generation_prompt", None)
    kwargs.pop("continue_final_message", None)
    effective_chat_template = (
        getattr(request, "chat_template", None) or chat_template
    )

    n = len(norm_messages)

    # Compute placeholder overhead once. The placeholder user message is
    # rendered by the Qwen template body loop unconditionally as
    # `<|im_start|>user\n{content}<|im_end|>\n` (chat_template.jinja:87-88).
    # Its contribution is INDEPENDENT of `tools` -- the tools-preamble is
    # emitted as a SEPARATE block before the body loop. So we probe with
    # tools=None to isolate just the user-message overhead. When the probe
    # is subtracted from a tools-aware injected-prefix render, the
    # tools-preamble cancels naturally:
    #     render([system, placeholder_user], tools=T)  =  T_with_sys + P
    #     - probe (tools=None)                         =          - P
    #     = T_with_sys                                 =  natural render([system], tools=T)
    # The probe MUST use add_generation_prompt=False to match the
    # placeholder rendering inside non-final prefixes (final prefix always
    # has a real user, so the placeholder branch never fires there).
    # Computed lazily (only once we see a prefix that needs it) to avoid
    # paying the probe cost for sessions that always have a user in messages[0].
    placeholder_overhead: int | None = None

    for i in range(1, n + 1):
        is_final = (i == n)
        # Engine uses request.add_generation_prompt (default True) on the
        # full conversation -- mirror that ONLY on the final iteration so
        # the gen_prompt overhead lands in the last range (preserves
        # cumulative-sum invariant).
        iter_add_gen = add_generation_prompt if is_final else False
        iter_continue_final = continue_final_message if is_final else False
        prefix = norm_messages[:i]
        if _prefix_has_user(prefix):
            rendered = tokenizer.apply_chat_template(
                prefix,
                chat_template=effective_chat_template,
                tools=tools,
                tokenize=True,
                add_generation_prompt=iter_add_gen,
                continue_final_message=iter_continue_final,
                **kwargs,
            )
            cur_len = len(rendered)
        else:
            if placeholder_overhead is None:
                # Probe with tools=None -- isolates the placeholder user
                # message's contribution (independent of tools-preamble).
                # Always add_generation_prompt=False -- placeholder branch
                # only fires for non-final prefixes (final always has a real
                # user message in mini-swe-agent flow).
                placeholder_overhead = len(tokenizer.apply_chat_template(
                    [dict(_PLACEHOLDER_USER_MSG)],
                    chat_template=effective_chat_template,
                    tools=None,
                    tokenize=True,
                    add_generation_prompt=False,
                    **kwargs,
                ))
            normalized = _inject_placeholder_after_leading_system(prefix)
            rendered = tokenizer.apply_chat_template(
                normalized,
                chat_template=effective_chat_template,
                tools=tools,
                tokenize=True,
                add_generation_prompt=iter_add_gen,
                continue_final_message=iter_continue_final,
                **kwargs,
            )
            cur_len = max(0, len(rendered) - placeholder_overhead)
        ranges.append((prev_len, cur_len))
        prev_len = cur_len
    return ranges


# =============================================================================
# SECTION C -- SessionTracker class
# =============================================================================


class SessionTracker:
    """Per-session live trajectory tracker (D-07).

    State (three indexed views over the same TurnState list):
      - file_timeline:    FileTimeline (D-07 view 1)
      - evictable_map:    EvictableSegmentMap (D-07 view 2; primary opportunity surface)
      - token_index:      TokenSequenceIndex (D-07 view 3; exact-match observation reuse)

    D-10 dynamic no-evict zone:
      - self.no_evict_zone_tokens: int | None
      - Initialized to None in __init__.
      - Computed on the session's first observe_request via
        _compute_initial_no_evict_zone (sum of leading system + first user
        message token-range lengths).
      - Locked thereafter. If a later request would re-derive a DIFFERENT
        value, the tracker logs a warning and keeps the original.

    Driver: observe_request processes ONLY new messages (D-13) and back-walks
    to mutate prior TurnStates when new evidence arrives (D-08). N-decay
    (D-09) re-applies after every observation.
    """

    def __init__(self, session_id: str, n_decay: int = 3) -> None:
        self.session_id = session_id
        self.n_decay = n_decay
        self.prev_msg_count = 0
        self.request_idx = 0
        self.file_timeline = FileTimeline()
        self.evictable_map = EvictableSegmentMap()
        self.token_index = TokenSequenceIndex()
        self.last_active = time.monotonic()
        # D-10: dynamic per-session no-evict zone. Computed on first
        # observe_request, locked thereafter. NOT a hardcoded constant -- the
        # legacy 1,991-token p95 value from Phase 01.1 is intentionally NOT
        # used by the live tracker.
        self.no_evict_zone_tokens: int | None = None
        # Cumulative stats for D-17 log line.
        self._cumulative_evictable_tokens = 0
        # Accumulated history (for re-classifier helpers that need full context).
        self._all_messages: list[dict] = []

    # ----- D-10 dynamic no-evict zone helper ----------------------------
    def _compute_initial_no_evict_zone(
        self,
        structured_messages: list[dict],
        message_token_ranges: list[tuple[int, int]],
    ) -> int:
        """D-10: sum token-range lengths of role=system messages plus the FIRST
        role=user message. Called only once, on the session's first
        observe_request invocation. mini-swe-agent appends only assistant/tool
        after the first user message, so the value is stable for the session.
        """
        first_user_seen = False
        total = 0
        for msg, (start, end) in zip(structured_messages, message_token_ranges):
            role = msg.get("role")
            if role == "system":
                total += (end - start)
            elif role == "user" and not first_user_seen:
                total += (end - start)
                first_user_seen = True
                break  # mini-swe-agent appends only assistant/tool after the first user message
        return total

    # ----- Public entry point -------------------------------------------
    def observe_request(
        self, *,
        structured_messages: list[dict],
        prompt_token_ids: list[int],
        message_token_ranges: list[tuple[int, int]],
    ) -> dict[str, Any]:
        """Process a single chat completion. Returns the D-17 log payload.

        On the session's first call, computes and locks the dynamic no-evict
        zone (D-10) from the leading system + first user messages. On
        subsequent calls, if the leading prefix has mutated the tracker logs
        a warning and keeps the original value.

        Mutates self.* in place. Idempotent over identical inputs (the
        message-count delta is the only state advancement signal -- if
        len(messages) == prev_msg_count, this is a no-op early return).
        """
        t0 = time.monotonic()
        self.request_idx += 1
        self.last_active = t0

        assert len(message_token_ranges) == len(structured_messages), (
            f"message_token_ranges length {len(message_token_ranges)} "
            f"must equal messages length {len(structured_messages)}"
        )
        if structured_messages:
            assert (
                len(prompt_token_ids) == 0
                or message_token_ranges[-1][1] == len(prompt_token_ids)
            ), (
                f"final range end {message_token_ranges[-1][1]} must equal "
                f"len(prompt_token_ids) {len(prompt_token_ids)} "
                "(RESEARCH Finding 7)"
            )

        # D-10: compute / verify the dynamic no-evict zone.
        if self.no_evict_zone_tokens is None:
            self.no_evict_zone_tokens = self._compute_initial_no_evict_zone(
                structured_messages, message_token_ranges,
            )
            logger.info(
                "agent_tracker: session %s no_evict_zone_tokens=%d (dynamic, locked)",
                self.session_id, self.no_evict_zone_tokens,
            )
        else:
            # If the leading prefix mutated, warn and keep the original.
            # Tolerance: the FIRST request's no_evict_zone may include the
            # add_generation_prompt scaffolding (~5 tokens for Qwen3.6) when
            # the first user message is also the last message in that request
            # -- subsequent requests don't include that scaffolding in the
            # range ending the first user. The benign +/-10-token drift is
            # purely from gen_prompt scaffolding, not actual content mutation;
            # don't spam logs for it. (Round 3 follow-on, 2026-04-28.)
            rederived = self._compute_initial_no_evict_zone(
                structured_messages, message_token_ranges,
            )
            drift = abs(rederived - self.no_evict_zone_tokens)
            if drift > 10:
                logger.warning(
                    "agent_tracker: leading prompt mutated mid-session for "
                    "sid=%s; keeping original no-evict zone=%d (would have been %d)",
                    self.session_id, self.no_evict_zone_tokens, rederived,
                )
            elif drift > 0:
                logger.debug(
                    "agent_tracker: tiny no-evict zone drift for sid=%s "
                    "(original=%d, rederived=%d, drift=%d <= 10 tokens, "
                    "likely add_generation_prompt scaffolding)",
                    self.session_id, self.no_evict_zone_tokens, rederived, drift,
                )

        new_msgs_count = len(structured_messages) - self.prev_msg_count
        new_evictable_tokens = 0
        exact_match_hit = False

        if new_msgs_count <= 0:
            # Idempotent no-op: same message count means no new evidence.
            payload = self._build_log_payload(
                new_msgs_count=0,
                new_evictable_tokens=0,
                exact_match_hit=False,
                latency_ms=0.0,
            )
            logger.info("agent_tracker: %s", payload)
            return payload

        # Replay through extract_new_messages on the FULL message list (the
        # offline algorithm groups consecutive new messages by role-pattern).
        prev_all = self._all_messages
        self._all_messages = list(structured_messages)
        # extract_new_messages expects "entries" with cumulative `messages` per
        # turn. We synthesize one entry whose messages are the full prior
        # context, then a second entry whose messages are the full new context.
        # The result is grouped[1] = the new messages of this request; we
        # combine with prior turn-grouping by re-running on the cumulative
        # turn-by-turn view via a minimal synthetic structure.
        #
        # In practice the live tracker tracks groups by walking the message
        # list in the same order the offline build_turn_states does: each
        # logical "turn group" corresponds to a contiguous block of new
        # messages between two natural boundaries (user task / assistant /
        # tool). For per-request incremental processing, we treat the entire
        # block of new messages as ONE turn group (matches offline's behavior
        # when extract_new_messages returns one entry per chat-completion
        # call). This preserves the parity with the offline classifier whose
        # `entries` are JSONL rows -- one per chat completion.
        new_msgs_block = structured_messages[self.prev_msg_count:]

        # --- BUILD NEW TURN(S) (incremental port of build_turn_states) ----
        # The offline algorithm processes one "entry" at a time; for the live
        # tracker, each observe_request call corresponds to one entry, so we
        # have at most ONE new turn-group per call (mirrors the JSONL row
        # structure that build_turn_states walks).
        if new_msgs_block:
            cumulative_msg_idx = self.prev_msg_count
            group_end = len(structured_messages)
            msg_range = (cumulative_msg_idx, group_end)
            token_range = (
                message_token_ranges[cumulative_msg_idx][0],
                message_token_ranges[group_end - 1][1],
            )
            turn_idx = len(self.evictable_map)

            # Reuse offline _classify_new_messages for category + command.
            category, command = _classify_new_messages(
                new_msgs_block, self._all_messages,
            )
            files_full = _extract_files_from_messages(
                new_msgs_block, prev_all,
            )
            files_basenames = {f.split("/")[-1] for f in files_full}

            # Token counts -- replace count_tokens with token_range slicing.
            token_count = token_range[1] - token_range[0]
            obs_tokens = (
                token_count if category in _OBSERVATION_CATEGORIES else 0
            )
            reasoning_tokens = (
                token_count if category == MSA_AGENT_REASONING else 0
            )
            tool_call_tokens = (
                token_count if category == MSA_AGENT_TOOL_CALL else 0
            )
            other_tokens = (
                token_count - obs_tokens - reasoning_tokens - tool_call_tokens
            )

            # Observation-message hash (D-11): hash the OBSERVATION message's
            # tokens. For tool/function role messages only.
            obs_hash: bytes | None = None
            for j in range(cumulative_msg_idx, group_end):
                role = self._all_messages[j].get("role")
                if role in ("tool", "function"):
                    seg_start, seg_end = message_token_ranges[j]
                    obs_hash = hash_token_sequence(
                        prompt_token_ids[seg_start:seg_end]
                    )
                    if obs_hash in self._all_obs_hashes_seen():
                        exact_match_hit = True
                    self.token_index.add(obs_hash, turn_idx)
                    break  # one obs per turn-group is sufficient

            ts = TurnState(
                turn_idx=turn_idx,
                category=category,
                files_referenced=files_basenames,
                files_referenced_full=files_full,
                command=command or None,
                is_edit=(category == MSA_TOOL_FILE_EDIT),
                is_success=True,  # offline default; refined by reclassification
                token_count=token_count,
                observation_tokens=obs_tokens,
                reasoning_tokens=reasoning_tokens,
                tool_call_tokens=tool_call_tokens,
                other_tokens=other_tokens,
                evictable=False,
                eviction_reason="essential",
                superseded_by=None,
                msg_range=msg_range,
                token_range=token_range,
                obs_token_hash=obs_hash,
            )
            self.evictable_map.append(ts)
            logger.debug(
                "agent_tracker: turn %d category=%s msg_range=%s token_range=%s",
                turn_idx, category, msg_range, token_range,
            )

            # Update FileTimeline (basename keys).
            action = self._action_from_category(category)
            for fp in files_full:
                bn = fp.split("/")[-1]
                self.file_timeline.append(
                    basename=bn, turn_idx=turn_idx, action=action,
                    msg_idx=cumulative_msg_idx, full_path=fp,
                )

            # --- D-08 RECLASSIFICATION: back-walk for this new turn -----
            self._reclassify_priors_for_new_turn(turn_idx)

        # --- D-09 N-DECAY (re-apply over ALL turns after new appends) ---
        self._apply_n_decay()

        # --- D-10 NO-EVICT ZONE FLOOR (final pass) ----------------------
        self._enforce_no_evict_zone_floor()

        # --- Stats -------------------------------------------------------
        # New evictable tokens = tokens marked evictable on turns added in
        # THIS request (i.e., turn indices >= the prior count of turns).
        # Since this driver appends at most one new turn per call, the count
        # of newly-added turns is `new_msgs_block` non-empty -> 1, else 0.
        new_turn_indices = []
        if new_msgs_block:
            new_turn_indices.append(len(self.evictable_map) - 1)
        new_evictable_tokens = sum(
            self.evictable_map[i].token_count
            for i in new_turn_indices
            if self.evictable_map[i].evictable
        )
        self._cumulative_evictable_tokens = sum(
            ts.token_count for ts in self.evictable_map if ts.evictable
        )

        self.prev_msg_count = len(structured_messages)
        latency_ms = (time.monotonic() - t0) * 1000.0

        payload = self._build_log_payload(
            new_msgs_count=new_msgs_count,
            new_evictable_tokens=new_evictable_tokens,
            exact_match_hit=exact_match_hit,
            latency_ms=latency_ms,
        )
        logger.info("agent_tracker: %s", payload)
        return payload

    # ----- Internal helpers ---------------------------------------------

    def _all_obs_hashes_seen(self) -> set[bytes]:
        return {
            ts.obs_token_hash
            for ts in self.evictable_map
            if ts.obs_token_hash is not None
        }

    def _action_from_category(self, category: str) -> str:
        return {
            MSA_TOOL_FILE_READ: "read",
            MSA_TOOL_FILE_EDIT: "edit",
            MSA_TOOL_FILE_SEARCH: "search",
            MSA_TOOL_TEST_RUN: "test",
            MSA_TOOL_BUILD_INSTALL: "build",
            MSA_TOOL_RUN_EXEC: "other",
            MSA_TOOL_OTHER: "other",
        }.get(category, "other")

    def _reclassify_priors_for_new_turn(self, new_turn_idx: int) -> None:
        """D-08: when a new turn references files, walk FileTimeline and mutate
        prior TurnStates' evictable/eviction_reason/superseded_by.

        Reuses _files_overlap from the offline classifier."""
        new_ts = self.evictable_map[new_turn_idx]
        if not new_ts.files_referenced:
            return
        for prior_idx in range(new_turn_idx):
            prior = self.evictable_map[prior_idx]
            if not prior.files_referenced:
                continue
            if _files_overlap(
                prior.files_referenced, new_ts.files_referenced,
                prior.files_referenced_full, new_ts.files_referenced_full,
            ):
                if new_ts.is_edit:
                    prior.evictable = True
                    prior.eviction_reason = "superseded_by_edit"
                    prior.superseded_by = new_turn_idx
                elif new_ts.category == MSA_TOOL_FILE_READ:
                    prior.evictable = True
                    prior.eviction_reason = "superseded_by_later_read"
                    prior.superseded_by = new_turn_idx

    def _apply_n_decay(self) -> None:
        """D-09: turns older than (n_turns - n_decay - 1) and not already
        evictable are marked decayed_N_turns. AGENT_TOOL_CALL turns are
        preserved as anchors per Phase 01.3 finding (`STATE.md`:
        'Do NOT evict Agent tool call records (28.2% budget) -- they are
        segment anchors')."""
        n = len(self.evictable_map)
        cutoff = n - self.n_decay - 1
        for ts in self.evictable_map:
            if ts.turn_idx > cutoff:
                continue
            if ts.evictable:
                continue
            if ts.category == MSA_AGENT_TOOL_CALL:
                continue
            if ts.category == MSA_SYSTEM_PROMPT or ts.category == MSA_USER_TASK:
                continue
            ts.evictable = True
            ts.eviction_reason = "decayed_N_turns"

    def _enforce_no_evict_zone_floor(self) -> None:
        """D-10: any TurnState whose token_range starts below the per-session
        no_evict_zone_tokens cannot be evictable, regardless of upstream
        classification. Uses self.no_evict_zone_tokens (DYNAMIC), NOT a
        hardcoded constant."""
        floor = self.no_evict_zone_tokens or 0
        for ts in self.evictable_map:
            if ts.token_range[0] < floor:
                ts.evictable = False
                ts.eviction_reason = "essential"
                ts.superseded_by = None

    def _build_log_payload(self, *, new_msgs_count: int,
                           new_evictable_tokens: int,
                           exact_match_hit: bool,
                           latency_ms: float) -> dict[str, Any]:
        """D-17 per-request structured log payload."""
        return {
            "session_id": self.session_id,
            "request_idx": self.request_idx,
            "new_msgs_count": new_msgs_count,
            "new_evictable_tokens_this_turn": new_evictable_tokens,
            "cumulative_evictable_tokens": self._cumulative_evictable_tokens,
            "exact_match_hit_this_turn": exact_match_hit,
            "latency_ms": round(latency_ms, 2),
        }

    def get_opportunity_dict(self) -> dict[str, Any]:
        """Build the D-16 OpportunityResponse JSON shape."""
        n_turns = len(self.evictable_map)
        total_tokens = sum(ts.token_count for ts in self.evictable_map)
        evictable_tokens = sum(
            ts.token_count for ts in self.evictable_map if ts.evictable
        )
        evictable_pct = (
            (100.0 * evictable_tokens / total_tokens) if total_tokens else 0.0
        )
        return {
            "session_id": self.session_id,
            "n_turns": n_turns,
            # D-10: dynamic per-session value, NOT the legacy constant.
            "no_evict_zone_tokens": self.no_evict_zone_tokens or 0,
            "evictable_token_total": evictable_tokens,
            "evictable_pct_of_total": round(evictable_pct, 2),
            "exact_match_turns": [
                {
                    "turn_idx": later,
                    "matches_turn_idx": earliest,
                    "token_count": self.evictable_map[later].token_count,
                }
                for (later, earliest) in self.token_index.exact_matches()
            ],
            "turns": [
                {
                    "turn_idx": ts.turn_idx,
                    "category": ts.category,
                    "evictable": ts.evictable,
                    "reason": ts.eviction_reason,
                    "msg_range": list(ts.msg_range),
                    "token_range": list(ts.token_range),
                    "superseded_by": ts.superseded_by,
                }
                for ts in self.evictable_map
            ],
            "file_timeline": self.file_timeline.to_dict(),
        }


# Observation categories (tool outputs); mirrors classifier._OBSERVATION_CATEGORIES
# but kept here to avoid a private-name import.
_OBSERVATION_CATEGORIES = {
    MSA_TOOL_FILE_READ,
    MSA_TOOL_FILE_EDIT,
    MSA_TOOL_FILE_SEARCH,
    MSA_TOOL_TEST_RUN,
    MSA_TOOL_BUILD_INSTALL,
    MSA_TOOL_RUN_EXEC,
    MSA_TOOL_OTHER,
}


# =============================================================================
# SECTION D -- SessionTrackerRegistry + module-level singleton accessor
# =============================================================================


class SessionTrackerRegistry:
    """LRU registry of SessionTrackers (D-14).

    Idle timeout + max-sessions cap. Process-global singleton; one instance
    per vLLM API server process. Use get_session_tracker_registry() (below)
    for access -- or stash on app.state.session_tracker_registry per
    Plan 04's init_app_state edit."""

    def __init__(self, *, idle_timeout_seconds: int = 1800,
                 max_sessions: int = 1000) -> None:
        self._sessions: OrderedDict[str, SessionTracker] = OrderedDict()
        self.idle_timeout = idle_timeout_seconds
        self.max_sessions = max_sessions

    def observe_request(self, *, session_id: str, **kwargs) -> dict:
        self._gc()
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionTracker(session_id)
            logger.info("agent_tracker: new session %s", session_id)
        self._sessions.move_to_end(session_id)
        return self._sessions[session_id].observe_request(**kwargs)

    def get_opportunity(self, session_id: str) -> dict | None:
        s = self._sessions.get(session_id)
        if s is None:
            return None
        return s.get_opportunity_dict()

    def delete(self, session_id: str) -> None:
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info("agent_tracker: deleted session %s", session_id)

    def __contains__(self, session_id: str) -> bool:
        return session_id in self._sessions

    def __len__(self) -> int:
        return len(self._sessions)

    def _gc(self) -> None:
        now = time.monotonic()
        for sid in list(self._sessions.keys()):
            if now - self._sessions[sid].last_active > self.idle_timeout:
                logger.info("agent_tracker: idle-evict session %s", sid)
                del self._sessions[sid]
        while len(self._sessions) >= self.max_sessions:
            oldest_sid, _ = self._sessions.popitem(last=False)
            logger.info("agent_tracker: lru-evict session %s", oldest_sid)


_REGISTRY: SessionTrackerRegistry | None = None


def get_session_tracker_registry() -> SessionTrackerRegistry:
    """Module-level accessor for the process-global registry.

    For tests / FastAPI integration: prefer Plan 04's stash on
    `app.state.session_tracker_registry`. This getter is the fallback
    when no app context is available."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = SessionTrackerRegistry()
    return _REGISTRY
