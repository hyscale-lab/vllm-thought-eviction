"""Line-span extraction for file-read/search observations.

The tracker still evicts whole tool-output observations, but supersession should
not treat "same file" as "same content". These helpers extract conservative
line spans from common shell/structured read tools so the tracker can only evict
an older read when a later read covers the same content.
"""
from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from vllm.agent_tracker.classifier import (
    _FILE_PATH_RE,
    TOOL_FILE_READ,
    TOOL_FILE_SEARCH,
    build_tool_call_command_map,
    build_tool_call_fn_name_map,
    classify_structured_tool,
)


@dataclass(frozen=True)
class FileReadSpan:
    path: str
    start_line: int | None = None
    end_line: int | None = None
    kind: str = "unknown"
    confidence: str = "unknown"

    @property
    def is_full_file(self) -> bool:
        return self.kind == "full_file"

    @property
    def has_exact_lines(self) -> bool:
        return (
            self.start_line is not None
            and self.end_line is not None
            and self.confidence == "exact"
        )


_SHELL_SPLIT_RE = re.compile(r"\s*(?:&&|\|\||;|\|)\s*")
_PATH_ARG_KEYS = (
    "path", "file", "filename", "file_path", "filepath", "target_file",
)
_START_KEYS = ("start_line", "line_start", "start", "line")
_END_KEYS = ("end_line", "line_end", "end")


def _shlex_split(text: str) -> list[str]:
    try:
        return shlex.split(text)
    except Exception:
        return text.split()


def _clean_path(token: str) -> str | None:
    if not token or token.startswith("-") or any(ch in token for ch in "<>"):
        return None
    if any(ch in token for ch in "*?[]"):
        return None
    token = token.strip("(){}")
    if not token or token.startswith("$("):
        return None
    if _FILE_PATH_RE.fullmatch(token):
        return token
    return None


def _path_tokens(tokens: list[str], *, skip_pattern_arg: bool = False) -> list[str]:
    out: list[str] = []
    skip_next = False
    pattern_skipped = False
    opts_with_arg = {
        "-n", "-e", "-f", "-m", "-A", "-B", "-C",
        "--max-count", "--after-context", "--before-context", "--context",
        "--exclude-dir", "--exclude", "--include",
    }
    for tok in tokens[1:]:
        if skip_next:
            skip_next = False
            continue
        if tok in opts_with_arg:
            skip_next = True
            continue
        if tok.startswith((
            "--exclude-dir=", "--exclude=", "--include=",
        )):
            continue
        if tok.startswith("-"):
            continue
        if skip_pattern_arg and not pattern_skipped:
            pattern_skipped = True
            continue
        path = _clean_path(tok)
        if path:
            out.append(path)
    return out


def _sed_range(tokens: list[str]) -> tuple[int, int] | None:
    if any(tok == "-i" or tok.startswith("-i") for tok in tokens[1:]):
        return None
    for tok in tokens[1:5] + tokens:
        match = re.fullmatch(r"(\d+)(?:,(\d+))?p", tok)
        if match is None:
            match = re.search(r"(\d+)(?:,(\d+))?p", tok)
        if match:
            start = int(match.group(1))
            end = int(match.group(2) or start)
            return (start, end)
    return None


def _head_range(tokens: list[str]) -> tuple[int | str, int | None]:
    if any(tok == "-c" or tok.startswith("-c") for tok in tokens):
        return ("prefix_bytes", None)
    n = 10
    for i, tok in enumerate(tokens):
        if tok == "-n" and i + 1 < len(tokens):
            try:
                n = int(tokens[i + 1])
            except ValueError:
                pass
        elif tok.startswith("-n") and len(tok) > 2:
            try:
                n = int(tok[2:])
            except ValueError:
                pass
        elif re.fullmatch(r"-\d+", tok):
            n = int(tok[1:])
    return (1, n)


def _grep_output_spans(stdout: str, command_paths: list[str]) -> list[FileReadSpan]:
    spans: list[FileReadSpan] = []
    command_path_set = set(command_paths)
    for line in stdout.splitlines():
        match = re.match(r"(.+?):(\d+)[:\-]", line)
        if not match:
            continue
        path = match.group(1)
        if command_path_set and not (
            path in command_path_set
            or any(path.startswith(p.rstrip("/") + "/") for p in command_path_set)
        ):
            continue
        line_no = int(match.group(2))
        spans.append(FileReadSpan(
            path=path, start_line=line_no, end_line=line_no,
            kind="line_match", confidence="exact",
        ))
    return _dedupe_spans(spans)


def _parse_command_spans(command: str, output: str = "") -> list[FileReadSpan]:
    spans: list[FileReadSpan] = []
    for raw_segment in _SHELL_SPLIT_RE.split(command or ""):
        tokens = _shlex_split(raw_segment.strip())
        if not tokens:
            continue
        head = tokens[0]
        if head in ("cd", "pushd", "popd", "pwd", "export", "source", "."):
            continue
        if head == "sed":
            line_range = _sed_range(tokens)
            if line_range is None:
                continue
            for path in _path_tokens(tokens):
                spans.append(FileReadSpan(
                    path=path, start_line=line_range[0], end_line=line_range[1],
                    kind="line_range", confidence="exact",
                ))
        elif head == "head":
            line_range = _head_range(tokens)
            for path in _path_tokens(tokens):
                if line_range[0] == "prefix_bytes":
                    spans.append(FileReadSpan(
                        path=path, kind="prefix_bytes", confidence="low",
                    ))
                else:
                    spans.append(FileReadSpan(
                        path=path, start_line=int(line_range[0]),
                        end_line=int(line_range[1] or line_range[0]),
                        kind="line_range", confidence="exact",
                    ))
        elif head in ("cat", "less", "more", "view", "nl", "tac"):
            if "<<" in raw_segment or re.search(r"(^|\s)cat\s*>", raw_segment):
                continue
            for path in _path_tokens(tokens):
                spans.append(FileReadSpan(
                    path=path, kind="full_file", confidence="full",
                ))
        elif head in ("grep", "rg", "ag", "ack"):
            command_paths = _path_tokens(tokens, skip_pattern_arg=True)
            match_spans = _grep_output_spans(output, command_paths)
            if match_spans:
                spans.extend(match_spans)
            else:
                for path in command_paths:
                    spans.append(FileReadSpan(
                        path=path, kind="unknown", confidence="unknown",
                    ))
        elif head == "tail":
            for path in _path_tokens(tokens):
                spans.append(FileReadSpan(
                    path=path, kind="unknown", confidence="unknown",
                ))
    return _dedupe_spans(spans)


def _int_arg(args: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        value = args.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _structured_spans_for_tool_call(tool_call: dict[str, Any]) -> list[FileReadSpan]:
    fn = tool_call.get("function") or {}
    if not isinstance(fn, dict):
        return []
    category = classify_structured_tool(str(fn.get("name") or ""))
    if category not in (TOOL_FILE_READ, TOOL_FILE_SEARCH):
        return []
    raw_args = fn.get("arguments") or {}
    if isinstance(raw_args, str):
        try:
            args = json.loads(raw_args)
        except Exception:
            args = {}
    else:
        args = raw_args
    if not isinstance(args, dict):
        return []
    path = next((str(args[k]) for k in _PATH_ARG_KEYS if args.get(k)), "")
    if not path:
        return []
    start = _int_arg(args, _START_KEYS)
    end = _int_arg(args, _END_KEYS)
    if start is not None:
        return [FileReadSpan(
            path=path, start_line=start, end_line=end or start,
            kind="line_range", confidence="exact",
        )]
    return [FileReadSpan(path=path, kind="full_file", confidence="full")]


def _tool_calls_by_id(messages: list[dict]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for tool_call in msg.get("tool_calls") or []:
            if not isinstance(tool_call, dict):
                continue
            tc_id = tool_call.get("id") or tool_call.get("tool_call_id")
            if tc_id:
                out[str(tc_id)] = tool_call
    return out


def _message_text(msg: dict[str, Any]) -> str:
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return ""


def extract_read_spans_from_messages(
    new_msgs: list[dict], all_msgs_so_far: list[dict],
) -> list[FileReadSpan]:
    """Extract conservative read/search spans for the new tool messages."""
    all_msgs = all_msgs_so_far + new_msgs
    command_map = build_tool_call_command_map(all_msgs)
    fn_map = build_tool_call_fn_name_map(all_msgs)
    calls_by_id = _tool_calls_by_id(all_msgs)
    spans: list[FileReadSpan] = []
    for msg in new_msgs:
        if msg.get("role") not in ("tool", "function"):
            continue
        tc_id = str(msg.get("tool_call_id", ""))
        command = command_map.get(tc_id, "")
        if command:
            spans.extend(_parse_command_spans(command, _message_text(msg)))
            continue
        category = classify_structured_tool(fn_map.get(tc_id, ""))
        if category in (TOOL_FILE_READ, TOOL_FILE_SEARCH):
            spans.extend(_structured_spans_for_tool_call(calls_by_id.get(tc_id, {})))
    return _dedupe_spans(spans)


def _same_path(a: str, b: str) -> bool:
    if a == b:
        return True
    pa = PurePosixPath(a)
    pb = PurePosixPath(b)
    return pa.name == pb.name and (
        str(pa).endswith(str(pb)) or str(pb).endswith(str(pa))
    )


def later_span_covers_prior(
    prior: FileReadSpan, later: FileReadSpan,
) -> bool:
    """True when dropping the whole prior observation is line-safe."""
    if not _same_path(prior.path, later.path):
        return False
    if later.is_full_file:
        return True
    if prior.is_full_file:
        return False
    if not (prior.has_exact_lines and later.has_exact_lines):
        return False
    return (
        later.start_line <= prior.start_line
        and later.end_line >= prior.end_line
    )



def _line_count(span: FileReadSpan) -> int | None:
    if not span.has_exact_lines:
        return None
    return span.end_line - span.start_line + 1


def later_span_is_narrower_than_prior(
    prior: FileReadSpan, later: FileReadSpan,
) -> bool:
    """True when a later exact read is a strict subset of a prior read.

    This is an aggressive policy signal, not a duplicate-content proof: it says
    the agent has narrowed attention from broad context to a smaller region.
    """
    if not _same_path(prior.path, later.path):
        return False
    if not later.has_exact_lines:
        return False
    if prior.is_full_file:
        return True
    if not prior.has_exact_lines:
        return False
    prior_lines = _line_count(prior)
    later_lines = _line_count(later)
    return (
        prior.start_line <= later.start_line
        and prior.end_line >= later.end_line
        and prior_lines is not None
        and later_lines is not None
        and later_lines < prior_lines
    )


def _dedupe_spans(spans: list[FileReadSpan]) -> list[FileReadSpan]:
    out: list[FileReadSpan] = []
    seen: set[FileReadSpan] = set()
    for span in spans:
        if span not in seen:
            seen.add(span)
            out.append(span)
    return out
