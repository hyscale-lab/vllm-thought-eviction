"""Trajectory classification primitives ported from Phase 01.3 offline
classifier (`scripts/trajectory_classifier.py`, `scripts/render_transcripts.py`,
`scripts/analyze_tokens.py`, `scripts/eviction_analysis.py`).

This module is PURE -- no I/O, no engine coupling, no async. The live
SessionTracker (`tracker.py`, Plan 03) imports these helpers and drives them
incrementally instead of the offline batch loop in `classify_conversation`.

NO-EVICT ZONE (D-10): the live tracker computes the no-evict zone DYNAMICALLY
per session at session start (sum of leading system + first user message
token ranges). The legacy 1,991-token p95 constant from
scripts/eviction_analysis.py is intentionally NOT ported here -- it remains
only in the offline curve script. K_DEFAULTS is the only constant lifted
from eviction_analysis.py.

Algorithm parity with the offline classifier is enforced by Plan 05's unit
tests (replay a Phase 01.1 JSONL through the tracker and diff against
`scripts/trajectory_classifier.classify_conversation` on the same
conversation).
"""
from __future__ import annotations

import json
import re
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# E-spec K-window defaults (ported from scripts/eviction_analysis.py:51-60).
#
# NOTE: the legacy 1,991-token p95 no-evict-zone constant is INTENTIONALLY
# NOT ported here. The live tracker computes the no-evict zone DYNAMICALLY
# per session per D-10 (sum of token-range lengths of leading system + first
# user messages). The legacy constant remains only in
# scripts/eviction_analysis.py for offline curve work.
# ---------------------------------------------------------------------------

# MSA category constants are defined below; K_DEFAULTS uses them as keys.
MSA_SYSTEM_PROMPT = "System prompt"
MSA_USER_TASK = "User task"
MSA_AGENT_REASONING = "Agent reasoning"
MSA_AGENT_TOOL_CALL = "Agent tool call"
MSA_TOOL_FILE_READ = "Tool output: file read"
MSA_TOOL_FILE_EDIT = "Tool output: file edit"
MSA_TOOL_FILE_SEARCH = "Tool output: file search"
MSA_TOOL_TEST_RUN = "Tool output: test run"
MSA_TOOL_BUILD_INSTALL = "Tool output: build/install"
MSA_TOOL_RUN_EXEC = "Tool output: run/exec"
MSA_TOOL_OTHER = "Tool output: other bash"

# Display order for the mini-swe-agent table
MSA_CATEGORY_ORDER = [
    MSA_SYSTEM_PROMPT,
    MSA_USER_TASK,
    MSA_AGENT_REASONING,
    MSA_AGENT_TOOL_CALL,
    MSA_TOOL_FILE_READ,
    MSA_TOOL_FILE_EDIT,
    MSA_TOOL_FILE_SEARCH,
    MSA_TOOL_TEST_RUN,
    MSA_TOOL_BUILD_INSTALL,
    MSA_TOOL_RUN_EXEC,
    MSA_TOOL_OTHER,
]

K_DEFAULTS = {
    MSA_TOOL_FILE_READ: 3,       # E-02
    MSA_TOOL_FILE_SEARCH: 3,     # E-03
    MSA_TOOL_RUN_EXEC: 3,        # E-04
    MSA_AGENT_REASONING: 5,      # E-05
    MSA_TOOL_BUILD_INSTALL: 2,   # E-06
    MSA_TOOL_OTHER: 3,           # E-07
    MSA_TOOL_TEST_RUN: 3,        # deferred but same as run/exec
    MSA_TOOL_FILE_EDIT: 999,     # effectively never evict (negligible volume)
}

# Compatibility aliases used by some downstream code that expects the
# normalized K_DEFAULTS shape from the plan frontmatter
# (TOOL_FILE_READ / TOOL_FILE_EDIT_DIFF / etc.). The canonical keys above use
# the MSA_* string constants because those are what _classify_new_messages /
# build_turn_states emit.
K_DEFAULTS["TOOL_FILE_READ"] = K_DEFAULTS[MSA_TOOL_FILE_READ]
K_DEFAULTS["TOOL_FILE_EDIT_DIFF"] = K_DEFAULTS[MSA_TOOL_FILE_EDIT]
K_DEFAULTS["TOOL_FILE_SEARCH"] = K_DEFAULTS[MSA_TOOL_FILE_SEARCH]
K_DEFAULTS["TOOL_TEST"] = K_DEFAULTS[MSA_TOOL_TEST_RUN]
K_DEFAULTS["TOOL_BUILD"] = K_DEFAULTS[MSA_TOOL_BUILD_INSTALL]
K_DEFAULTS["TOOL_BASH_OTHER"] = K_DEFAULTS[MSA_TOOL_OTHER]
K_DEFAULTS["AGENT_REASONING"] = K_DEFAULTS[MSA_AGENT_REASONING]
K_DEFAULTS["AGENT_TOOL_CALL"] = 0  # tool-call turns are segment anchors, never evicted


# ---------------------------------------------------------------------------
# File-path extraction primitives (ported from scripts/render_transcripts.py:499-602).
# ---------------------------------------------------------------------------

# Use non-capturing group for extension alternatives so findall returns the full match.
#
# URL-rejection guards (added 2026-04-28 per quick task 260428-w4j):
#   - Negative lookbehind `(?<![/:\w])` on both alternatives rejects matches whose
#     previous char is `:`, `/`, or word char. This kills URL-context fragments
#     like `://`, `host/path`, and `:password` that previously polluted
#     agent_tracker_live.json::file_timeline (e.g. `//foo`, `//userid`, `/pass`).
#   - `(?:/|~/)` (instead of `[/~]`) requires `~` to be followed by `/` so that
#     URL special-char strings like `http://-.~_!...@example.com` no longer
#     yield bare `~_` matches. `~/foo` and `~/projects/x.py` still match.
_FILE_PATH_RE = re.compile(
    r'(?<![/:\w])(?:/|~/)[\w./\-]+'
    r'|(?<![/:\w])[\w./\-]+\.(?:py|js|ts|java|go|rb|sh|md|txt|json|yaml|yml|toml|cfg|conf|cpp|c|h)(?!\w)'
)

# Tool names that carry bash/shell commands.
_BASH_TOOL_RE = re.compile(r'bash|execute|run', re.IGNORECASE)


def extract_turn_signals(rows):
    """Return a list of per-turn dicts with keys: file_reads, bash_commands, code_blocks.

    file_reads: list of filenames / paths found in tool call arguments.
    bash_commands: list of shell command strings (first 200 chars) from bash-like tool calls.
    code_blocks: list of stripped non-empty code-fence contents (first 300 chars) from assistant text.
    """
    signals = []
    for row in rows:
        resp = row.get("response") or {}
        if not isinstance(resp, dict):
            resp = {}
        assistant_content = resp.get("content") or ""
        tool_calls = resp.get("tool_calls") or []
        if not isinstance(tool_calls, list):
            tool_calls = []

        file_reads = set()
        bash_commands = []
        seen_commands = set()

        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function") or {}
            if not isinstance(fn, dict):
                fn = {}
            fn_name = fn.get("name") or ""
            fn_args_raw = fn.get("arguments") or ""
            if isinstance(fn_args_raw, (dict, list)):
                fn_args_str = json.dumps(fn_args_raw)
            else:
                fn_args_str = str(fn_args_raw)

            # File reads: extract from any tool call arguments.
            for match in _FILE_PATH_RE.findall(fn_args_str):
                file_reads.add(match)

            # Bash commands: only from bash-like tool calls.
            if _BASH_TOOL_RE.search(fn_name):
                cmd = None
                try:
                    parsed = json.loads(fn_args_str)
                    if isinstance(parsed, dict):
                        cmd = parsed.get("command") or parsed.get("cmd") or parsed.get("input")
                except (json.JSONDecodeError, TypeError, ValueError):
                    pass
                if cmd is None:
                    cmd = fn_args_str
                if cmd is not None:
                    cmd = str(cmd).strip()[:200]
                    if cmd and cmd not in seen_commands:
                        bash_commands.append(cmd)
                        seen_commands.add(cmd)

        # Code blocks from assistant text. Store FULL content (not truncated) so that
        # subset matching in build_similarity_report can detect cases where a shorter
        # block is contained within a longer one from another turn. Display-time
        # truncation happens in render_similarity_report.
        code_blocks = []
        seen_blocks = set()
        if assistant_content:
            for block in re.findall(r'```[^\n]*\n(.*?)```', assistant_content, re.DOTALL):
                stripped = block.strip()
                if stripped and stripped not in seen_blocks:
                    code_blocks.append(stripped)
                    seen_blocks.add(stripped)

        signals.append({
            "file_reads": list(file_reads),
            "bash_commands": bash_commands,
            "code_blocks": code_blocks,
        })
    return signals


def _is_path_suffix(shorter, longer):
    """True if `shorter` is a path suffix of `longer` at a path-component boundary.

    Used for file-read subset matching so that 'bar.py' groups with '/foo/bar.py'
    (same file via different path forms) while 'src' does NOT group with
    'src/main.py' (not a suffix, just a shared prefix component).

    Examples:
      _is_path_suffix('bar.py', '/foo/bar.py')        -> True
      _is_path_suffix('main.py', 'src/main.py')       -> True
      _is_path_suffix('src', 'src/main.py')           -> False (prefix, not suffix)
      _is_path_suffix('foo.py', 'bar/foo.pyc')        -> False (extension mismatch)
    """
    if not shorter or not longer or len(shorter) >= len(longer):
        return False
    if not longer.endswith(shorter):
        return False
    boundary = longer[-len(shorter) - 1]
    return boundary in ('/', '\\')


# ---------------------------------------------------------------------------
# Mini-swe-agent bash command classifier (ported from
# scripts/analyze_tokens.py:393-651).
#
# Heuristic priority: search > read > test > build > edit > run > other.
# ---------------------------------------------------------------------------


def build_tool_call_command_map(messages: list) -> dict:
    """Map tool_call_id -> bash command for mini-swe-agent re-categorization.

    Walks assistant messages, extracts tool_calls[i].id and the 'command' key from
    its JSON-encoded arguments. Used by categorize_message_parts_minisweagent to
    re-classify role:tool messages based on the command that produced them.
    """
    mapping: dict = {}
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for tc in (msg.get("tool_calls") or []):
            tc_id = tc.get("id") or tc.get("tool_call_id")
            if not tc_id:
                continue
            fn = tc.get("function") or {}
            args_raw = fn.get("arguments") or "{}"
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            except Exception:
                args = {}
            command = ""
            if isinstance(args, dict):
                command = args.get("command", "") or ""
            mapping[tc_id] = command
    return mapping


def build_tool_call_fn_name_map(messages: list) -> dict:
    """Map tool_call_id -> tool function name.

    Companion to build_tool_call_command_map for agents (e.g. Hermes) that do
    file I/O through STRUCTURED tools (`read`/`grep`/`edit` with a `path` arg)
    rather than bash `cat`/`grep` commands. Used to categorize a tool turn by
    function name when there is no bash command to parse.
    """
    mapping: dict = {}
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for tc in (msg.get("tool_calls") or []):
            tc_id = tc.get("id") or tc.get("tool_call_id")
            if not tc_id:
                continue
            fn = tc.get("function") or {}
            mapping[tc_id] = str(fn.get("name") or "").strip().lower()
    return mapping


# Structured-tool function-name -> intent. Checked in EDIT, SEARCH, READ order so
# e.g. "rewrite"/"str_replace" (contain "write"/"replace") classify as edit, not
# read, and "file_search" classifies as search. Names vary across agents, so both
# exact tokens and substrings are matched. Bash/exec tools are intentionally
# absent: they carry a `command` and go through classify_bash_command_minisweagent.
_STRUCT_EDIT_KEYS = ("str_replace", "replace", "write", "edit", "patch", "insert",
                     "create_file", "createfile", "modify", "apply_diff", "apply_patch")
_STRUCT_SEARCH_KEYS = ("grep", "ripgrep", "glob", "search", "find", "locate",
                       "list_dir", "list_files", "listdir", "tree")
_STRUCT_READ_KEYS = ("read", "view", "cat", "open_file", "openfile", "show_file",
                     "get_file", "getfile", "fileread", "read_file")


def classify_structured_tool(fn_name: str) -> str:
    """Categorize a structured (non-bash) tool call by its function name.

    Returns one of MSA_TOOL_FILE_{EDIT,SEARCH,READ} or "" when the name is not a
    recognized file tool (caller then falls back to MSA_TOOL_OTHER).
    """
    n = (fn_name or "").strip().lower()
    if not n:
        return ""
    if any(k in n for k in _STRUCT_EDIT_KEYS):
        return MSA_TOOL_FILE_EDIT
    if any(k in n for k in _STRUCT_SEARCH_KEYS):
        return MSA_TOOL_FILE_SEARCH
    if any(k in n for k in _STRUCT_READ_KEYS):
        return MSA_TOOL_FILE_READ
    return ""


_MSA_FILE_READ_HEADS = {"cat", "head", "tail", "less", "more", "view", "nl", "tac"}
_MSA_FILE_SEARCH_HEADS = {"find", "grep", "rg", "ag", "ack", "ls", "which",
                          "locate", "whereis", "tree"}
_MSA_RUN_EXEC_HEADS = {"node", "ruby", "perl", "lua", "java", "deno", "bun"}
_MSA_TEST_HEADS = {"pytest", "py.test", "tox", "nosetests", "jest", "mocha"}
_MSA_BUILD_HEADS = {"cmake", "ninja", "bazel", "buck", "mvn", "gradle"}


def _msa_strip_envs(tokens: list) -> list:
    """Drop leading VAR=value tokens (env-var prefixes) before head detection."""
    i = 0
    while i < len(tokens) and "=" in tokens[i] and not tokens[i].startswith(("-", "/")):
        # Heuristic: VAR=value has no slash before the '='
        head = tokens[i].split("=", 1)[0]
        if head and head[0].isalpha() and head.replace("_", "").isalnum():
            i += 1
            continue
        break
    return tokens[i:]


def _msa_first_head(command: str) -> tuple:
    """Return (head, tokens_after_head) of the first sub-command in a chained line.

    Splits on shell operators (&&, ||, ;, |) and returns the head token of the
    first non-trivial sub-command (skipping `cd` prefixes which are navigation,
    not the intent). Strips env-var assignments. The returned head is the
    *intent* of the line for classification purposes.
    """
    if not command:
        return ("", [])
    # Split on shell operators (very rough but good enough for classification)
    chunks = re.split(r"(?:&&|\|\||;|\|)", command)
    for raw in chunks:
        toks = raw.strip().split()
        if not toks:
            continue
        toks = _msa_strip_envs(toks)
        if not toks:
            continue
        head = toks[0]
        # Skip pure navigation / no-op prefixes (cd, pwd, env-setters, etc.).
        # These are scaffolding; the real intent of the line is the next chunk.
        if head in (
            "cd", "pushd", "popd", "pwd", "set", "export", "unset", "true",
            ":", "source", ".", "umask", "alias", "ulimit",
        ):
            continue
        return (head, toks[1:])
    # All chunks were skips -- fall back to the very first token
    toks = command.strip().split()
    toks = _msa_strip_envs(toks)
    if toks:
        return (toks[0], toks[1:])
    return ("", [])


def classify_bash_command_minisweagent(command: str, content: str) -> str:
    """Classify a bash command into one of the 7 MSA tool-output sub-buckets.

    See module-level comment in scripts/analyze_tokens.py for heuristic priority
    and edge cases. `content` is accepted for symmetry with
    `_classify_bash_observation` and to allow future content-aware refinements
    (e.g. detecting `>500` chars edit echoes as reads), but the v1
    implementation classifies purely on the command.
    """
    head, rest = _msa_first_head((command or "").strip())
    if not head:
        return MSA_TOOL_OTHER

    # 1. File search (must come BEFORE file-read so `grep -r` doesn't get
    #    miscategorized as a read of the search-pattern argument).
    if head in _MSA_FILE_SEARCH_HEADS:
        return MSA_TOOL_FILE_SEARCH

    # 2. File read
    if head in _MSA_FILE_READ_HEADS:
        return MSA_TOOL_FILE_READ
    if head == "sed" and rest and rest[0].startswith("-n"):
        return MSA_TOOL_FILE_READ

    # 3. Test runners
    if head in _MSA_TEST_HEADS:
        return MSA_TOOL_TEST_RUN
    if head in ("python", "python3", "python2"):
        # `python -m pytest`, `python -m unittest`, `python -m nose`
        if len(rest) >= 2 and rest[0] == "-m" and rest[1] in (
            "pytest", "unittest", "nose", "nose2", "tox"
        ):
            return MSA_TOOL_TEST_RUN
        # `python -m pip install` -> build/install
        if len(rest) >= 3 and rest[0] == "-m" and rest[1] == "pip" and "install" in rest[2:]:
            return MSA_TOOL_BUILD_INSTALL
        # everything else python -> run/exec
        return MSA_TOOL_RUN_EXEC

    # 4. Build / install
    if head in _MSA_BUILD_HEADS:
        return MSA_TOOL_BUILD_INSTALL
    if head == "pip" and rest and rest[0] in ("install", "uninstall", "wheel"):
        return MSA_TOOL_BUILD_INSTALL
    if head in ("conda", "mamba", "micromamba") and rest and rest[0] in ("install", "create", "update"):
        return MSA_TOOL_BUILD_INSTALL
    if head in ("apt", "apt-get", "yum", "dnf", "pacman", "brew") and rest and rest[0] in ("install", "update", "upgrade"):
        return MSA_TOOL_BUILD_INSTALL
    if head == "make":
        # `make test`/`make check` -> test_run; everything else (incl. bare `make`,
        # `make install`) is build_install.
        if rest and rest[0] in ("test", "tests", "check", "checks"):
            return MSA_TOOL_TEST_RUN
        return MSA_TOOL_BUILD_INSTALL
    if head in ("npm", "yarn", "pnpm"):
        if rest and rest[0] in ("test", "run") and len(rest) >= 2 and rest[1] in ("test", "tests"):
            return MSA_TOOL_TEST_RUN
        if rest and rest[0] == "test":
            return MSA_TOOL_TEST_RUN
        if rest and rest[0] in ("install", "i", "add", "ci"):
            return MSA_TOOL_BUILD_INSTALL
        return MSA_TOOL_RUN_EXEC
    if head == "cargo":
        if rest and rest[0] in ("test", "bench"):
            return MSA_TOOL_TEST_RUN
        if rest and rest[0] in ("build", "install"):
            return MSA_TOOL_BUILD_INSTALL
        if rest and rest[0] == "run":
            return MSA_TOOL_RUN_EXEC
        return MSA_TOOL_BUILD_INSTALL
    if head == "go":
        if rest and rest[0] == "test":
            return MSA_TOOL_TEST_RUN
        if rest and rest[0] in ("build", "install", "get", "mod"):
            return MSA_TOOL_BUILD_INSTALL
        if rest and rest[0] == "run":
            return MSA_TOOL_RUN_EXEC

    # 5. File edit (after build to avoid `make install > log` matching `>`)
    if head == "sed" and rest and any(r.startswith("-i") for r in rest):
        return MSA_TOOL_FILE_EDIT
    if head in ("patch", "tee"):
        return MSA_TOOL_FILE_EDIT
    if head in ("mv", "cp", "rm", "mkdir", "touch", "ln", "chmod", "chown"):
        return MSA_TOOL_FILE_EDIT
    # echo/printf into a file via shell redirect -- head is echo/printf and the
    # full command contains `>` or `>>`
    if head in ("echo", "printf") and (" > " in command or " >> " in command):
        return MSA_TOOL_FILE_EDIT
    # heredoc patterns: `cat <<EOF > file` writes a file
    if head == "cat" and ("<<" in command and (" > " in command or " >> " in command)):
        return MSA_TOOL_FILE_EDIT

    # 6. Run / exec (interpreters not caught above)
    if head in _MSA_RUN_EXEC_HEADS:
        return MSA_TOOL_RUN_EXEC
    if head.startswith("./") or head.startswith("/"):
        # Direct binary execution
        return MSA_TOOL_RUN_EXEC
    if head in ("bash", "sh", "zsh", "ksh"):
        return MSA_TOOL_RUN_EXEC

    # 7. Default
    return MSA_TOOL_OTHER


# ---------------------------------------------------------------------------
# Trajectory classification helpers (ported from
# scripts/trajectory_classifier.py:86-254, 449-467).
# ---------------------------------------------------------------------------


def extract_new_messages(entries: list[dict]) -> list[list[dict]]:
    """For each turn, extract only the NEW messages added since the prior turn.

    entries: list of traj rows for a single conversation, ordered by turn.
    Returns: list of message lists, one per turn, containing only the new messages.

    Each traj row's `messages` is cumulative. New messages at turn N =
    row[N].messages[len(row[N-1].messages):] (Pitfall 2 avoidance).
    """
    result = []
    prev_msg_count = 0
    for entry in entries:
        messages = entry.get("messages", [])
        new_msgs = messages[prev_msg_count:]
        result.append(new_msgs)
        prev_msg_count = len(messages)
    return result


# Categories that are always essential
_ALWAYS_ESSENTIAL = {MSA_SYSTEM_PROMPT, MSA_USER_TASK, MSA_AGENT_TOOL_CALL}

# Observation categories (tool outputs)
_OBSERVATION_CATEGORIES = {
    MSA_TOOL_FILE_READ, MSA_TOOL_FILE_EDIT, MSA_TOOL_FILE_SEARCH,
    MSA_TOOL_TEST_RUN, MSA_TOOL_BUILD_INSTALL, MSA_TOOL_RUN_EXEC,
    MSA_TOOL_OTHER,
}


def _classify_new_messages(new_msgs: list[dict], all_msgs_so_far: list[dict]) -> tuple[str, str]:
    """Determine the primary category of new messages and extract the bash command.

    Returns (category, command).
    """
    if not new_msgs:
        return MSA_AGENT_REASONING, ""

    # Check what roles are present
    roles = [m.get("role") for m in new_msgs]

    # If system message present, it's the system prompt turn
    if "system" in roles:
        return MSA_SYSTEM_PROMPT, ""

    # If user message present (and no assistant), it's user task
    if "user" in roles and "assistant" not in roles:
        return MSA_USER_TASK, ""

    # If tool message present, classify based on the command that produced it
    if "tool" in roles or "function" in roles:
        # Find the tool message
        for msg in new_msgs:
            if msg.get("role") in ("tool", "function"):
                tc_id = msg.get("tool_call_id", "")
                content = msg.get("content", "")
                if isinstance(content, list):
                    content = " ".join(
                        item.get("text", "") if isinstance(item, dict) else str(item)
                        for item in content
                    )
                # Find the command for this tool_call_id
                all_msgs = all_msgs_so_far + new_msgs
                command = build_tool_call_command_map(all_msgs).get(tc_id, "")
                if command:
                    category = classify_bash_command_minisweagent(command, content)
                    return category, command
                # Structured tool with no bash command (e.g. Hermes read/grep/edit
                # carrying a `path` arg): categorize by the tool function name so
                # file reads are recognized and can be superseded/evicted.
                struct = classify_structured_tool(
                    build_tool_call_fn_name_map(all_msgs).get(tc_id, ""))
                if struct:
                    return struct, ""
                return MSA_TOOL_OTHER, ""
        return MSA_TOOL_OTHER, ""

    # If assistant message with tool_calls, it's an agent tool call
    for msg in new_msgs:
        if msg.get("role") == "assistant":
            if msg.get("tool_calls"):
                return MSA_AGENT_TOOL_CALL, ""
            # Check for </think> marker
            text = msg.get("content", "")
            if isinstance(text, list):
                text = " ".join(
                    item.get("text", "") if isinstance(item, dict) else str(item)
                    for item in text
                )
            if "</think>" in text:
                # Has both reasoning and potentially tool call parts
                _, after = text.split("</think>", 1)
                if after.strip():
                    return MSA_AGENT_TOOL_CALL, ""
            return MSA_AGENT_REASONING, ""

    return MSA_AGENT_REASONING, ""


def _extract_files_from_messages(new_msgs: list[dict], all_msgs: list[dict]) -> set[str]:
    """Extract file paths referenced in these messages.

    Uses extract_turn_signals for tool call arguments and also parses
    tool message content for file paths.
    """
    files = set()

    # Build a synthetic row for extract_turn_signals
    # extract_turn_signals expects rows with 'response' having 'content' and 'tool_calls'
    for msg in new_msgs:
        if msg.get("role") == "assistant":
            row = {"response": msg}
            signals = extract_turn_signals([row])
            if signals:
                for f in signals[0].get("file_reads", []):
                    files.add(f)

    # Also extract file paths from tool messages and commands
    cmd_map = build_tool_call_command_map(all_msgs + new_msgs)
    for msg in new_msgs:
        if msg.get("role") in ("tool", "function"):
            tc_id = msg.get("tool_call_id", "")
            command = cmd_map.get(tc_id, "")
            # Extract file paths from the command itself
            if command:
                for match in _FILE_PATH_RE.findall(command):
                    if match:
                        files.add(match)

    return files


def _files_overlap(files_a: set[str], files_b: set[str],
                   files_a_full: set[str], files_b_full: set[str]) -> bool:
    """Check if two sets of files overlap using normalized names and path suffix matching.

    Uses both exact basename matching and _is_path_suffix for cross-reference detection.
    """
    # Direct normalized match
    if files_a & files_b:
        return True

    # Path suffix matching on full paths
    for fa in files_a_full:
        for fb in files_b_full:
            if fa == fb:
                return True
            if _is_path_suffix(fa, fb) or _is_path_suffix(fb, fa):
                return True

    return False
