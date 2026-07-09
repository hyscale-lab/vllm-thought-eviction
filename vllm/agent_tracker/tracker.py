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

import threading
import time
from collections import OrderedDict
from collections.abc import Iterable
from typing import Any

from vllm.logger import init_logger

from vllm.agent_tracker.classifier import (
    AGENT_REASONING,
    AGENT_TOOL_CALL,
    TOOL_FILE_READ,
    TOOL_FILE_EDIT,
    TOOL_FILE_SEARCH,
    TOOL_TEST_RUN,
    TOOL_BUILD_INSTALL,
    TOOL_RUN_EXEC,
    TOOL_OTHER,
    SYSTEM_PROMPT,
    USER_TASK,
    OBSERVATION_CATEGORIES,
    DEDUPE_OUTPUT_CATEGORIES,
    _MIN_DEDUPE_CHARS,
    _classify_new_messages,
    _extract_files_from_messages,
    _files_overlap,
    normalize_observation_text,
)
from vllm.agent_tracker.file_timeline import FileTimeline
from vllm.agent_tracker.segment_map import EvictableSegmentMap, TurnState
from vllm.agent_tracker.token_index import (
    TokenSequenceIndex,
    hash_text,
    hash_token_sequence,
)

logger = init_logger(__name__)


# =============================================================================
# SECTION B -- compute_message_token_ranges helper
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
        # NORMALIZATION (c) -- mirror the engine's reasoning handling.
        # vLLM drops assistant `reasoning_content` when it parses chat messages
        # into the prompt (the field is not threaded through parse_chat_messages),
        # so the Qwen template renders an EMPTY `<think>\n\n</think>` block for
        # historical assistant turns. apply_chat_template here is fed the raw
        # request messages, which still carry reasoning_content, so it would
        # render the full reasoning text -- inflating each assistant turn's range
        # by its reasoning length and tripping the Finding-7 assertion
        # (ranges[-1][1] != len(prompt_token_ids)) on ~90% of Hermes turns, which
        # disables eviction. Blank it so the tracker's render matches the engine's
        # prompt_token_ids exactly. (Hermes debug 2026-06-04: engine=5119 vs
        # raw-render=5158 == the turn's 39 reasoning tokens; cumulative per turn.)
        if msg.get("role") == "assistant" and msg.get("reasoning_content"):
            msg["reasoning_content"] = ""
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


def _message_text(msg: Any) -> str:
    """Best-effort flatten of a chat message's content to text for cheap
    equality checks. Handles str content and OpenAI-style list-of-parts
    content; falls back to str() for anything else."""
    if not isinstance(msg, dict):
        return ""
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, dict):
                out.append(str(part.get("text", part.get("content", ""))))
            else:
                out.append(str(part))
        return "".join(out)
    return "" if content is None else str(content)


def _framing_signature(messages: list) -> tuple:
    """Signature of a conversation's IMMUTABLE framing prefix: the leading
    role=system block plus the FIRST role=user message. Coding agents (Hermes /
    OpenClaw) hold this constant across a conversation while appending only
    assistant/tool turns, so the signature is identical on every turn of a given
    conversation but differs between distinct ones. Used to detect a session_id
    reused by a genuinely different conversation (see SessionTracker.is_diverged).
    """
    parts: list[tuple[str, str]] = []
    for m in messages:
        if not isinstance(m, dict):
            break
        role = m.get("role")
        if role == "system":
            parts.append(("system", _message_text(m)))
        elif role == "user":
            parts.append(("user", _message_text(m)))
            break
        else:
            # assistant/tool before any user -> no stable framing to anchor on.
            break
    return tuple(parts)


# HF fast (Rust) tokenizers are not thread-safe: concurrent encode/apply_chat_
# template borrows raise "Already borrowed" (observed as HTTP 500 on 13/24 and
# 7/24 simultaneous tracker requests in the 2026-07-02 loadtests -- the failures
# recur whenever many sessions' rounds align, not just at cold start). The
# engine tokenizes on its own path; this lock only serializes the TRACKER's
# re-render, whose cost is small next to prefill.
_TRACKER_TOKENIZE_LOCK = threading.Lock()


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
    start_from: int = 0,
) -> list[tuple[int, int]]:
    with _TRACKER_TOKENIZE_LOCK:
        return _compute_message_token_ranges_unlocked(
            messages, tokenizer, request, chat_template,
            chat_template_content_format, default_chat_template_kwargs,
            tools=tools, add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            start_from=start_from,
        )


def _compute_message_token_ranges_unlocked(
    messages: list,
    tokenizer,
    request,
    chat_template: str | None,
    chat_template_content_format: str,
    default_chat_template_kwargs: dict,
    tools: list[dict] | None = None,
    add_generation_prompt: bool = True,
    continue_final_message: bool = False,
    start_from: int = 0,
) -> list[tuple[int, int]]:
    """Per-message token ranges via incremental re-tokenization.

    When ``start_from > 0``, skips the first ``start_from`` prefixes and fills
    those positions with ``(0, 0)`` sentinels.  A single boundary
    ``apply_chat_template(messages[:start_from])`` call seeds ``prev_len``,
    then only the remaining ``N - start_from`` prefixes are tokenized.  This
    reduces per-request cost from O(N) tokenizer calls to O(new_msgs) --
    typically 3-4 calls instead of 100+ for long conversations.

    Callers MUST treat ``ranges[j]`` for ``j < start_from`` as invalid.

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
        messages[:1] = [system]) always trips this for a system-first agent. We
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
    start_from = max(0, min(start_from, n))

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

    def _rendered_len(render_msgs, *, agp, cfm, tools_arg):
        """Token length of a rendered prefix, the SAME way the vLLM engine
        builds prompt_token_ids: render to a string (tokenize=False) and then
        encode with add_special_tokens=False (the chat template emits BOS
        itself; see ChatCompletionRequest.add_special_tokens default False).

        We deliberately do NOT use apply_chat_template(tokenize=True): its
        fast path is broken for some custom chat templates whose markers are
        not in the tokenizer's added-tokens. The Gemma4 template
        (<|turn>/<|tool_call>/<|tool_response>/<|"|> ...) is the motivating
        case -- tokenize=True collapses the WHOLE render to ~2 tokens (they
        round-trip on decode but encode to far more), which zeroed every
        per-message range and made evictable_token_total=0. Render-then-encode
        matches the engine for Gemma AND is identical to the old tokenize=True
        length for well-behaved templates (Qwen), so the framing assertion
        ranges[-1][1] == len(prompt_token_ids) keeps holding for both.
        """
        text = tokenizer.apply_chat_template(
            render_msgs,
            chat_template=effective_chat_template,
            tools=tools_arg,
            tokenize=False,
            add_generation_prompt=agp,
            continue_final_message=cfm,
            **kwargs,
        )
        return len(tokenizer.encode(text, add_special_tokens=False))

    def _render_prefix(prefix, *, is_final):
        nonlocal placeholder_overhead
        agp = add_generation_prompt if is_final else False
        cfm = continue_final_message if is_final else False
        if _prefix_has_user(prefix):
            return _rendered_len(prefix, agp=agp, cfm=cfm, tools_arg=tools)
        if placeholder_overhead is None:
            placeholder_overhead = _rendered_len(
                [dict(_PLACEHOLDER_USER_MSG)],
                agp=False, cfm=False, tools_arg=None,
            )
        normalized = _inject_placeholder_after_leading_system(prefix)
        rendered_len = _rendered_len(
            normalized, agp=agp, cfm=cfm, tools_arg=tools,
        )
        return max(0, rendered_len - placeholder_overhead)

    # -- Fast path: skip prefixes [1..start_from] via a single boundary call --
    if start_from > 0:
        ranges: list[tuple[int, int]] = [(0, 0)] * start_from
        prev_len = _render_prefix(norm_messages[:start_from], is_final=False)
        loop_start = start_from + 1
    else:
        ranges = []
        prev_len = 0
        loop_start = 1

    for i in range(loop_start, n + 1):
        cur_len = _render_prefix(norm_messages[:i], is_final=(i == n))
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

    def __init__(
        self, session_id: str, n_decay: int | None = None,
        dedupe_cmd_output: bool = False,
        supersede_reads: bool = True,
    ) -> None:
        self.session_id = session_id
        # Time-decay is OFF by default (None): _apply_n_decay short-circuits, so
        # eviction is driven by supersede_reads + dedupe_cmd_output unless an arm
        # opts into a decay window. An int enables the round tail-guard window.
        self.n_decay = n_decay
        # Ablation knob, locked per-session at tracker creation (like n_decay):
        # content-hash dedupe of repeated run/exec + other-bash output.
        self.dedupe_cmd_output = dedupe_cmd_output
        # Ablation knob, locked per-session at tracker creation (like n_decay):
        # file-overlap supersession (superseded_by_later_read / _by_edit) in
        # _reclassify_priors_for_new_turn. Default ON (legacy behavior); set
        # False to run N-decay in isolation.
        self.supersede_reads = supersede_reads
        self.prev_msg_count = 0
        self.request_idx = 0
        self.file_timeline = FileTimeline()
        self.evictable_map = EvictableSegmentMap()
        self.token_index = TokenSequenceIndex()
        # Content-hash dedupe of repeated command output (findings §5): maps the
        # normalized-text digest of a run/exec / other-bash observation to the
        # turn indices that produced it, so a later identical-after-normalization
        # rerun can supersede the earlier copies.
        self._norm_hash_index: dict[bytes, list[int]] = {}
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
        # Per-tool eviction: monotonic round counter. Incremented once per
        # assistant message (one agent action) in the per-message turn build;
        # every TurnState created for that action shares the round. N-decay
        # counts these rounds rather than raw turn indices.
        self._round_counter = 0
        # Per-request authoritative server-side prefill log (one entry per engine
        # request): the prefill BEFORE eviction (raw) and what the engine actually
        # computes AFTER the transparent drop (post). Serialized into the
        # opportunity dict so the analysis can build an exact per-request
        # prefill CDF instead of reconstructing it from evicted_at_round.
        self._prefill_requests: list[dict] = []

    # ----- Robustness: reused/reset session_id detection ----------------
    def is_diverged(self, structured_messages: list) -> bool:
        """True if ``structured_messages`` is NOT an append-extension of what
        this tracker last observed -- i.e. the client replaced the conversation
        under the same session_id. The incremental fast-path (start_from =
        prev_msg_count - 1) and all persisted turn state assume each request
        only APPENDS messages; when that breaks the tracker must reset (the hook
        calls registry.should_reset -> registry.reset).

        Two triggers:

        (a) SHRINK -- the request has fewer messages than we've already
            processed. The dominant cause in practice is a client-side context
            compaction: the agent's prompt overflows the model context window
            (vLLM returns 400), and the harness (e.g. Hermes) restarts/compacts
            the conversation into a much smaller message list under the SAME
            session_id. Compaction typically preserves the system + first-user
            framing, so (b) would NOT catch it -- the message-count drop is the
            reliable signal.

        (b) FRAMING MUTATION -- same-or-greater message count but the immutable
            leading system + first-user framing changed, meaning a different
            conversation reused this session_id.
        """
        if len(structured_messages) < self.prev_msg_count:
            return True
        if self._all_messages and self.prev_msg_count > 0:
            return (
                _framing_signature(structured_messages)
                != _framing_signature(self._all_messages)
            )
        return False

    # ----- D-10 dynamic no-evict zone helper ----------------------------
    def _compute_initial_no_evict_zone(
        self,
        structured_messages: list[dict],
        message_token_ranges: list[tuple[int, int]],
    ) -> int:
        """D-10: sum token-range lengths of role=system messages plus the FIRST
        role=user message. Called only once, on the session's first
        observe_request invocation. The agent appends only assistant/tool
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
                break  # the agent appends only assistant/tool after the first user message
        return total

    # ----- Public entry point -------------------------------------------
    def observe_request(
        self, *,
        structured_messages: list[dict],
        prompt_token_ids: list[int],
        message_token_ranges: list[tuple[int, int]],
        partial_ranges: bool = False,
    ) -> dict[str, Any]:
        """Process a single chat completion. Returns the D-17 log payload.

        On the session's first call, computes and locks the dynamic no-evict
        zone (D-10) from the leading system + first user messages. On
        subsequent calls, if the leading prefix has mutated the tracker logs
        a warning and keeps the original value.

        When ``partial_ranges`` is True, early entries in
        ``message_token_ranges`` are ``(0, 0)`` sentinels (see
        ``compute_message_token_ranges(start_from=...)``).  The no-evict
        zone drift check is skipped because the sentinel ranges would
        produce a spurious mismatch.

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
        # The incremental per-message render in compute_message_token_ranges
        # under-counts a TRAILING run of consecutive `tool` messages: Qwen merges
        # them into a single <tool_response> block whose close
        # (`</tool_response><|im_end|>`) and the add_generation_prompt suffix
        # (`<|im_start|>assistant<think>`) fall outside every per-message prefix diff.
        # That left ranges[-1][1] < len(prompt_token_ids), which used
        # to trip a HARD assert and drop EVERY post-tool turn (n_turns stuck 0).
        # Instead, snap the final message's range end up to the engine length so
        # cumulative accounting is exact. The snapped residual is structural
        # scaffolding (+ merge-boundary slack); it is recorded in the turn's
        # token_range but kept OUT of the evictable obs span -- see the
        # `_snap_unsnapped_end` cap in the obs_token_range loop below. An
        # OVER-count (ranges claim more than the engine produced) is a genuine
        # alignment bug, not this undercount, so it keeps the hard guard.
        _snap_unsnapped_end: int | None = None
        if structured_messages and prompt_token_ids:
            _last_start, _last_end = message_token_ranges[-1]
            _engine_len = len(prompt_token_ids)
            if _last_end < _engine_len:
                message_token_ranges = list(message_token_ranges)
                _snap_unsnapped_end = _last_end
                message_token_ranges[-1] = (_last_start, _engine_len)
                logger.debug(
                    "agent_tracker: snapped final range %d->%d for sid=%s "
                    "(trailing tool-block scaffolding; +%d tokens recorded in "
                    "token_range but excluded from obs span)",
                    _last_end, _engine_len, self.session_id,
                    _engine_len - _last_end,
                )
            else:
                assert _last_end == _engine_len, (
                    f"final range end {_last_end} exceeds "
                    f"len(prompt_token_ids) {_engine_len}"
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
        elif not partial_ranges:
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

        # Accumulate full history for the re-classifier / is_diverged helpers.
        self._all_messages = list(structured_messages)
        new_msgs_block = structured_messages[self.prev_msg_count:]
        new_turn_indices: list[int] = []

        # --- PER-MESSAGE TURN BUILD (per-tool eviction, 2026-06-23) ---------
        # Emit one TurnState per NEW message instead of collapsing the whole
        # [assistant, tool, tool, ...] block into a single turn. Each
        # tool/function message becomes its OWN independently-evictable
        # observation (own obs_token_range / obs_hash / files); each assistant
        # tool-call becomes an always-essential anchor that opens a new decay
        # ROUND. Reuses the offline per-message classifier + file extractor;
        # downstream supersession / decay / serialization stay per-TurnState.
        if new_msgs_block:
            # Cross-request boundary trim (unchanged): the prior request rendered
            # its final message WITH the +gen-prompt suffix; this request renders
            # that same message non-finally, exposing the natural end. Trim the
            # prior request's last turn so persisted ranges don't overlap. Skip
            # on a LARGE backward jump (a last_query_index frame rebase, not a
            # scaffolding adjustment) to keep the prior range internally monotonic.
            if self.prev_msg_count > 0 and self.evictable_map:
                prev_ts = self.evictable_map[-1]
                natural_end = message_token_ranges[self.prev_msg_count - 1][1]
                old_start, old_end = prev_ts.token_range
                if natural_end < old_end and natural_end >= old_start:
                    delta = old_end - natural_end
                    prev_ts.token_range = (old_start, natural_end)
                    prev_ts.token_count = max(0, prev_ts.token_count - delta)
                    for fld in ("observation_tokens", "reasoning_tokens",
                                "tool_call_tokens", "other_tokens"):
                        v = getattr(prev_ts, fld)
                        if v >= delta:
                            setattr(prev_ts, fld, v - delta)
                            break

            final_idx = len(structured_messages) - 1
            for j in range(self.prev_msg_count, len(structured_messages)):
                msg = structured_messages[j]
                role = msg.get("role")
                seg_start, seg_end = message_token_ranges[j]

                # Per-message non-monotonic guard: a chat-template
                # last_query_index flip can make this prefix shorter than the
                # previous one (end < start). Skip THIS turn but keep walking so
                # later messages still chain forward in the post-flip frame.
                if seg_end < seg_start:
                    logger.warning(
                        "agent_tracker: skipping non-monotonic msg turn for "
                        "sid=%s msg_idx=%d range=(%d,%d) (last_query_index flip)",
                        self.session_id, j, seg_start, seg_end,
                    )
                    continue

                # Round bookkeeping: each assistant message is a new agent
                # action -> a new round; tool results inherit the current round.
                if role == "assistant":
                    self._round_counter += 1
                cur_round = self._round_counter

                # Classify just this message (single-element block) so each tool
                # gets its OWN category/command/files resolved by tool_call_id.
                prefix = structured_messages[:j]
                one = [msg]
                category, command = _classify_new_messages(one, prefix)
                files_full = _extract_files_from_messages(one, prefix)
                files_basenames = {f.split("/")[-1] for f in files_full}

                token_range = (seg_start, seg_end)
                token_count = seg_end - seg_start
                obs_tokens = (
                    token_count if category in OBSERVATION_CATEGORIES else 0
                )
                reasoning_tokens = (
                    token_count if category == AGENT_REASONING else 0
                )
                tool_call_tokens = (
                    token_count if category == AGENT_TOOL_CALL else 0
                )
                other_tokens = (
                    token_count - obs_tokens - reasoning_tokens - tool_call_tokens
                )

                turn_idx = len(self.evictable_map)
                obs_hash: bytes | None = None
                obs_token_range: tuple[int, int] | None = None
                obs_norm_hash: bytes | None = None
                if role in ("tool", "function"):
                    o_start, o_end = seg_start, seg_end
                    # Keep the snapped structural residual OUT of the evictable
                    # obs span when this obs IS the snapped final message (the
                    # snap inflates only message_token_ranges[final_idx]).
                    if _snap_unsnapped_end is not None and j == final_idx:
                        o_end = _snap_unsnapped_end
                    obs_token_range = (o_start, o_end)
                    obs_hash = hash_token_sequence(
                        prompt_token_ids[o_start:o_end]
                    )
                    if obs_hash in self._all_obs_hashes_seen():
                        exact_match_hit = True
                    self.token_index.add(obs_hash, turn_idx)
                    # Content-hash dedupe key (run/exec + other-bash only):
                    # hash the NORMALIZED observation text so reruns that differ
                    # only in timing/addresses/tmp-paths collapse to one digest.
                    if (
                        self.dedupe_cmd_output
                        and category in DEDUPE_OUTPUT_CATEGORIES
                    ):
                        norm = normalize_observation_text(_message_text(msg))
                        if len(norm) >= _MIN_DEDUPE_CHARS:
                            obs_norm_hash = hash_text(norm)

                ts = TurnState(
                    turn_idx=turn_idx,
                    category=category,
                    files_referenced=files_basenames,
                    files_referenced_full=files_full,
                    command=command or None,
                    is_edit=(category == TOOL_FILE_EDIT),
                    is_success=True,
                    token_count=token_count,
                    observation_tokens=obs_tokens,
                    reasoning_tokens=reasoning_tokens,
                    tool_call_tokens=tool_call_tokens,
                    other_tokens=other_tokens,
                    evictable=False,
                    eviction_reason="essential",
                    superseded_by=None,
                    msg_range=(j, j + 1),
                    token_range=token_range,
                    obs_token_range=obs_token_range,
                    obs_token_hash=obs_hash,
                    obs_norm_hash=obs_norm_hash,
                    round_idx=cur_round,
                )
                self.evictable_map.append(ts)
                new_turn_indices.append(turn_idx)
                logger.debug(
                    "agent_tracker: turn %d round=%d role=%s category=%s "
                    "token_range=%s obs=%s", turn_idx, cur_round, role,
                    category, token_range, obs_token_range,
                )

                # Update FileTimeline (basename keys).
                action = self._action_from_category(category)
                for fp in files_full:
                    bn = fp.split("/")[-1]
                    self.file_timeline.append(
                        basename=bn, turn_idx=turn_idx, action=action,
                        msg_idx=j, full_path=fp,
                    )

                # D-08 supersession back-walk: an earlier observation can be
                # superseded by a later read even within this same request.
                self._reclassify_priors_for_new_turn(turn_idx)

                # Content-hash dedupe of repeated command output (no-op unless
                # this session enabled dedupe_cmd_output).
                self._dedupe_repeated_output(turn_idx)

        # --- D-09 N-DECAY (re-apply over ALL turns after new appends) ---
        self._apply_n_decay()

        # --- D-10 NO-EVICT ZONE FLOOR (final pass) ----------------------
        self._enforce_no_evict_zone_floor()

        # --- Stats -------------------------------------------------------
        # New evictable tokens = tokens marked evictable on turns added in THIS
        # request (per-message build appends one turn per new message; their
        # indices were collected in new_turn_indices above).
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
            TOOL_FILE_READ: "read",
            TOOL_FILE_EDIT: "edit",
            TOOL_FILE_SEARCH: "search",
            TOOL_TEST_RUN: "test",
            TOOL_BUILD_INSTALL: "build",
            TOOL_RUN_EXEC: "other",
            TOOL_OTHER: "other",
        }.get(category, "other")

    def _reclassify_priors_for_new_turn(self, new_turn_idx: int) -> None:
        """D-08: when a new turn references files, walk FileTimeline and mutate
        prior TurnStates' evictable/eviction_reason/superseded_by.

        Reuses _files_overlap from the offline classifier.

        D-19 PARITY (debug round 4, 2026-04-28): two corrections vs the
        original implementation so that incremental back-walking matches
        offline `classify_conversation`'s forward-scan semantics
        (`scripts/trajectory_classifier.py:526-539`):

        (1) FIRST supersedor wins. The offline forward scan from turn i
            breaks on the first overlap with a target later turn; later
            supersedors do NOT overwrite the attribution. The live
            incremental back-walk must NOT overwrite a prior turn's
            already-set supersession reason. EXCEPTION: provisional
            `decayed_N_turns` marks (which `_apply_n_decay` may have set
            speculatively before the real supersedor arrived) MUST be
            allowed to upgrade to a `superseded_by_*` reason -- offline
            decides supersession-vs-decay only AFTER seeing the whole
            conversation, so the live tracker has to upgrade decay-marks
            when stronger evidence appears later.

        (2) FILE_SEARCH is a supersedor too. The offline classifier
            treats FILE_READ + FILE_SEARCH symmetrically as
            `superseded_by_later_read` (line 535). The original live
            implementation silently dropped SEARCH, leaving many turns
            falling through to `decayed_N_turns` instead of
            `superseded_by_later_read=<search_idx>`.
        """
        # Ablation gate: when supersede_reads is disabled, ALL file-overlap
        # supersession below is a no-op -- both superseded_by_later_read (a
        # later read/search of a file) AND superseded_by_edit (a later edit,
        # which supersedes the now-stale earlier reads of that file). Eviction
        # is then driven only by whatever else the arm enabled: N-decay (opt-in
        # via AGENT_TRACKER_N_DECAY; OFF by default) and/or dedupe_cmd_output
        # (its own gate).
        if not self.supersede_reads:
            return
        new_ts = self.evictable_map[new_turn_idx]
        if not new_ts.files_referenced:
            return
        for prior_idx in range(new_turn_idx):
            prior = self.evictable_map[prior_idx]
            # Per-tool eviction: only OBSERVATION turns are actually evictable
            # (serving.py drops obs_token_range). Anchors / reasoning / user /
            # system turns have obs_token_range=None and drop nothing, so never
            # flag them evictable -- it would only inflate evictable_pct.
            if prior.obs_token_range is None:
                continue
            if not prior.files_referenced:
                continue
            # Correction (1): preserve the FIRST (earliest) supersedor.
            # Provisional decay marks may still be upgraded.
            if prior.evictable and prior.eviction_reason != "decayed_N_turns":
                continue
            if _files_overlap(
                prior.files_referenced, new_ts.files_referenced,
                prior.files_referenced_full, new_ts.files_referenced_full,
            ):
                if new_ts.is_edit:
                    prior.evictable = True
                    prior.eviction_reason = "superseded_by_edit"
                    prior.superseded_by = new_turn_idx
                elif new_ts.category in (
                    TOOL_FILE_READ, TOOL_FILE_SEARCH,
                ):
                    # Correction (2): SEARCH is symmetric with READ for
                    # supersession (mirror offline line 535).
                    prior.evictable = True
                    prior.eviction_reason = "superseded_by_later_read"
                    prior.superseded_by = new_turn_idx

    def _dedupe_repeated_output(self, new_turn_idx: int) -> None:
        """Content-hash dedupe of repeated command output (findings doc §5).

        Path-based supersession (`_reclassify_priors_for_new_turn`) only catches
        re-reads/edits of the same FILE. The large un-deduped leak is re-running
        the same script / test / `ls` whose OUTPUT is near-identical but is not
        file content, so path overlap never fires; those copies survive on
        time-decay alone. This method closes that gap for run/exec + other-bash
        turns: when a new turn's normalized observation text matches an earlier
        such turn's, the earlier copy is provably redundant and is marked
        `superseded_by_repeat` + evictable, mirroring the `superseded_by_later_read`
        path so the engine prefill filter, analysis, and viz pick it up.

        Only the EARLIER copies are evicted (the most-recent rerun is kept), the
        same keep-latest semantics as later-read supersession. Parity with
        `_reclassify_priors_for_new_turn`'s correction (1): an existing
        `superseded_by_*` attribution is preserved (first supersedor wins), while
        `essential` and provisional `decayed_N_turns` marks are (re)attributed to
        the repeat so the engine filter actually drops them. No-op unless the
        session enabled ``dedupe_cmd_output``.
        """
        if not self.dedupe_cmd_output:
            return
        new_ts = self.evictable_map[new_turn_idx]
        if new_ts.category not in DEDUPE_OUTPUT_CATEGORIES:
            return
        nh = new_ts.obs_norm_hash
        if nh is None:
            return
        for prior_idx in self._norm_hash_index.get(nh, ()):
            prior = self.evictable_map[prior_idx]
            if prior.evictable and prior.eviction_reason in (
                "superseded_by_later_read",
                "superseded_by_edit",
                "superseded_by_repeat",
            ):
                continue
            prior.evictable = True
            prior.eviction_reason = "superseded_by_repeat"
            prior.superseded_by = new_turn_idx
        # Register this turn as the newest producer of `nh` so the next rerun
        # supersedes it in turn.
        self._norm_hash_index.setdefault(nh, []).append(new_turn_idx)

    def _apply_n_decay(self) -> None:
        """
        ROUND-AWARE (per-tool eviction): once each tool message is
        its own turn, a single agent action (assistant + N parallel tool
        results) spans N+1 turns. Counting the decay window in RAW turns would
        let one parallel-tool burst exhaust the lookahead. So the window and the
        tail guard count ROUNDS (TurnState.round_idx) instead -- preserving the
        pre-split horizon (one request ~ one round).

        For each turn that survived supersession (not already evictable):
          - Obs gate: only turns with an obs_token_range are actually evictable
            (eviction drops obs_token_range); skip anchors/reasoning.
          - Tail guard: spare turns within the last n_decay ROUNDS.
          - File-window check: if the turn references files AND any of those
            files appears in a LATER round within n_decay rounds, do NOT decay.
          - No-files obs turns (e.g. test runs) decay purely by round count.
        """
        # Time-decay disabled (default): eviction is supersession/dedupe-driven.
        if self.n_decay is None:
            return
        n = len(self.evictable_map)
        states = self.evictable_map.all_turns()
        n_decay = self.n_decay
        max_round = states[-1].round_idx if states else 0
        for i, ts in enumerate(states):
            if ts.evictable:
                continue
            if ts.category == AGENT_TOOL_CALL:
                continue
            if ts.category in (SYSTEM_PROMPT, USER_TASK):
                continue
            # Only turns with an actual observation can be evicted (serving.py
            # drops obs_token_range); reasoning/anchor turns drop nothing.
            if ts.obs_token_range is None:
                continue
            # Round-aware tail guard: spare turns within the last n_decay rounds.
            if ts.round_idx + n_decay > max_round:
                continue
            if (
                ts.category not in OBSERVATION_CATEGORIES
                and ts.category != AGENT_REASONING
            ):
                continue
            if ts.files_referenced:
                referenced_in_window = False
                for j in range(i + 1, n):
                    later = states[j]
                    # Same-round siblings are the SAME action, not a future
                    # reference; stop once we pass the n_decay-round window.
                    if later.round_idx <= ts.round_idx:
                        continue
                    if later.round_idx > ts.round_idx + n_decay:
                        break
                    if _files_overlap(
                        ts.files_referenced, later.files_referenced,
                        ts.files_referenced_full,
                        later.files_referenced_full,
                    ):
                        referenced_in_window = True
                        break
                if not referenced_in_window:
                    ts.evictable = True
                    ts.eviction_reason = "decayed_N_turns"
            else:
                # No files referenced -- pure round-decay (e.g. test runs).
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

    def mark_evicted(self, turn_indices: Iterable[int]) -> list[int]:
        """Record that ``turn_indices`` were dropped from the engine prefill on
        the CURRENT request and return the rounds newly recorded (one per turn
        whose ``evicted_at_round`` had not been set yet), for metrics.

        The eviction round is the session's CURRENT round (the round_idx of the
        most recently observed turn). First drop wins: a decayed turn is dropped
        on every subsequent request, but ``evicted_at_round`` captures the FIRST
        round it left the context. Turns already recorded are skipped so the
        metric counts each section once.
        """
        states = self.evictable_map.all_turns()
        if not states:
            return []
        cur_round = states[-1].round_idx
        newly: list[int] = []
        for ti in turn_indices:
            if 0 <= ti < len(states):
                ts = states[ti]
                if ts.evicted_at_round is None:
                    ts.evicted_at_round = cur_round
                    newly.append(cur_round)
        return newly

    def record_prefill(self, raw_token_count: int, filtered_count: int) -> dict:
        """Log the raw and post-eviction prefill token counts for the CURRENT
        request. Called by the serving eviction hook after it computes the drop
        (filtered_count == 0 when nothing was dropped), so the opportunity JSON
        carries an exact per-request prefill trace for the analysis CDF."""
        states = self.evictable_map.all_turns()
        cur_round = states[-1].round_idx if states else 0
        rec = {
            "request_idx": self.request_idx,
            "round_idx": cur_round,
            "raw_prefill_tokens": int(raw_token_count),
            "filtered_tokens": int(filtered_count),
            "post_evict_prefill_tokens": int(raw_token_count) - int(filtered_count),
        }
        self._prefill_requests.append(rec)
        return rec

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
                    "obs_token_range": (
                        list(ts.obs_token_range) if ts.obs_token_range else None
                    ),
                    "superseded_by": ts.superseded_by,
                    "round_idx": ts.round_idx,
                    "evicted_at_round": ts.evicted_at_round,
                }
                for ts in self.evictable_map
            ],
            "file_timeline": self.file_timeline.to_dict(),
            # Authoritative per-request server-side prefill trace (option C);
            # empty when no request has been recorded yet.
            "prefill_requests": list(self._prefill_requests),
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
        # Per-session prefill trace that SURVIVES tracker resets. The
        # tracker-local SessionTracker._prefill_requests is wiped on every
        # reset() -- and some harnesses (e.g. OpenClaw) reset on nearly every
        # request because their per-request messages are not a clean token-level
        # append, so only the last request's entry would survive there. Keyed on
        # session_id here, the trace accumulates across resets for the SAME
        # conversation; a leading-framing change (a genuinely different
        # conversation reusing a session_id) starts a fresh trace. Bounded by the
        # same delete()/_gc() lifecycle as _sessions.
        self._prefill_log: dict[str, list[dict]] = {}
        self._prefill_framing: dict[str, tuple] = {}

    def get_prev_msg_count(self, session_id: str) -> int:
        """Return the session's prev_msg_count (0 if session doesn't exist yet)."""
        s = self._sessions.get(session_id)
        return s.prev_msg_count if s is not None else 0

    def observe_request(
        self, *, session_id: str, n_decay: int | None = None,
        dedupe_cmd_output: bool | None = None,
        supersede_reads: bool | None = None, **kwargs
    ) -> dict:
        self._gc()
        if session_id not in self._sessions:
            # n_decay / dedupe_cmd_output / supersede_reads (if provided by the
            # client) are per-session knobs, locked at tracker creation; later
            # requests reuse the existing tracker.
            init_kwargs: dict = {}
            if n_decay is not None:
                init_kwargs["n_decay"] = n_decay
            if dedupe_cmd_output is not None:
                init_kwargs["dedupe_cmd_output"] = dedupe_cmd_output
            if supersede_reads is not None:
                init_kwargs["supersede_reads"] = supersede_reads
            self._sessions[session_id] = SessionTracker(session_id, **init_kwargs)
            logger.info(
                "agent_tracker: new session %s (n_decay=%s, dedupe_cmd_output=%s, "
                "supersede_reads=%s)",
                session_id, self._sessions[session_id].n_decay,
                self._sessions[session_id].dedupe_cmd_output,
                self._sessions[session_id].supersede_reads,
            )
        self._sessions.move_to_end(session_id)
        return self._sessions[session_id].observe_request(**kwargs)

    def should_reset(self, session_id: str, structured_messages: list) -> bool:
        """True if an existing tracker for ``session_id`` has diverged from
        ``structured_messages`` (reused/compacted/restarted conversation). The
        hook consults this BEFORE computing token ranges so it can drop the
        stale tracker and recompute from scratch (start_from=0) instead of
        emitting (0,0)-sentinel ranges. Returns False for unknown sessions."""
        s = self._sessions.get(session_id)
        return s is not None and s.is_diverged(structured_messages)

    def reset(self, session_id: str) -> None:
        """Drop any tracker for ``session_id`` so the next observe_request
        rebuilds it fresh. Used by the hook when a session_id is reused for a
        new/compacted conversation."""
        # NB: intentionally does NOT touch _prefill_log/_prefill_framing -- the
        # per-request prefill trace must accumulate ACROSS resets for the same
        # conversation (that is the whole point of keying it on the registry).
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info(
                "agent_tracker: reset session %s (conversation replaced/"
                "compacted -- rebuilding tracker)", session_id,
            )

    def get_opportunity(self, session_id: str) -> dict | None:
        s = self._sessions.get(session_id)
        if s is None:
            return None
        d = s.get_opportunity_dict()
        # Prefer the reset-surviving session-keyed trace over the tracker-local
        # _prefill_requests (which, after a reset, holds only this request).
        if session_id in self._prefill_log:
            d["prefill_requests"] = list(self._prefill_log[session_id])
        return d

    def mark_evicted(self, session_id: str, turn_indices) -> list[int]:
        """Record which turns server-side eviction dropped this request; returns
        the rounds newly recorded (for metrics). No-op for unknown sessions."""
        s = self._sessions.get(session_id)
        if s is None:
            return []
        return s.mark_evicted(turn_indices)

    def record_prefill(self, session_id: str, raw_token_count: int,
                       filtered_count: int) -> None:
        """Append this request's raw/post-eviction prefill to the SESSION-keyed
        trace (``_prefill_log``) that survives tracker resets. No-op for unknown
        sessions. A change in the leading system + first-user framing (a
        genuinely different conversation reusing the session_id) starts a fresh
        trace; otherwise records accumulate across the per-request resets of the
        same conversation, with request_idx renumbered monotonically."""
        s = self._sessions.get(session_id)
        if s is None:
            return
        framing = _framing_signature(s._all_messages)
        if self._prefill_framing.get(session_id) != framing:
            self._prefill_log[session_id] = []
            self._prefill_framing[session_id] = framing
        log = self._prefill_log[session_id]
        rec = s.record_prefill(raw_token_count, filtered_count)
        rec["request_idx"] = len(log) + 1  # monotonic across resets
        log.append(rec)

    def delete(self, session_id: str) -> None:
        if session_id in self._sessions:
            del self._sessions[session_id]
            logger.info("agent_tracker: deleted session %s", session_id)
        self._prefill_log.pop(session_id, None)
        self._prefill_framing.pop(session_id, None)

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
                self._prefill_log.pop(sid, None)
                self._prefill_framing.pop(sid, None)
        while len(self._sessions) >= self.max_sessions:
            oldest_sid, _ = self._sessions.popitem(last=False)
            logger.info("agent_tracker: lru-evict session %s", oldest_sid)
            self._prefill_log.pop(oldest_sid, None)
            self._prefill_framing.pop(oldest_sid, None)


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
