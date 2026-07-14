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
only in the offline curve script.

Algorithm parity with the offline classifier is enforced by Plan 05's unit
tests (replay a Phase 01.1 JSONL through the tracker and diff against
`scripts/trajectory_classifier.classify_conversation` on the same
conversation).
"""
from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from vllm.logger import init_logger

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Trajectory category constants. Agent-agnostic: they label the segments of a
# coding-agent conversation (Hermes / OpenClaw) regardless of harness.
#
# NOTE: the legacy 1,991-token p95 no-evict-zone constant is INTENTIONALLY
# NOT defined here. The live tracker computes the no-evict zone DYNAMICALLY
# per session per D-10 (sum of token-range lengths of leading system + first
# user messages). The legacy constant remains only in the offline curve script.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = "System prompt"
USER_TASK = "User task"
AGENT_REASONING = "Agent reasoning"
AGENT_TOOL_CALL = "Agent tool call"
TOOL_FILE_READ = "Tool output: file read"
TOOL_FILE_EDIT = "Tool output: file edit"
TOOL_FILE_SEARCH = "Tool output: file search"
TOOL_TEST_RUN = "Tool output: test run"
TOOL_BUILD_INSTALL = "Tool output: build/install"
TOOL_RUN_EXEC = "Tool output: run/exec"
TOOL_WEB_SEARCH = "Tool output: web search"
TOOL_WEB_FETCH = "Tool output: web fetch"
TOOL_OTHER = "Tool output: other bash"


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

# URL extraction for web_search/web_fetch eviction (D-20): pulls http(s) links
# out of a web_search tool's result content or a web_fetch call's arguments.
# Trailing punctuation that a sentence/markdown wrapper tacks on (`.`, `)`,
# `"`, closing markdown `]`) is stripped by `_normalize_url` below, not here,
# so the regex itself stays a plain "greedy till whitespace/bracket" match.
_URL_RE = re.compile(r'https?://[^\s<>\)\]"\']+')


def _normalize_url(url: str) -> str:
    """Normalize a URL for overlap comparisons across search results and
    fetch targets: strips trailing sentence/markdown punctuation and the
    fragment, lowercases scheme+host, and drops a trailing slash on the path.

    Two URLs that normalize to the same string are treated as "the same
    page" for eviction purposes (e.g. a web_fetch of `HTTP://Example.com/x/`
    consumes the search result `https://example.com/x`).
    """
    stripped = (url or "").strip().rstrip('.,;:!?)"\']')
    if not stripped:
        return ""
    try:
        parts = urlsplit(stripped)
    except ValueError:
        return stripped
    path = parts.path.rstrip('/')
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, parts.query, ""))


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
# Bash command classifier (ported from
# scripts/analyze_tokens.py:393-651).
#
# Heuristic priority: search > read > test > build > edit > run > other.
# ---------------------------------------------------------------------------


def build_tool_call_command_map(messages: list) -> dict:
    """Map tool_call_id -> bash command for tool-output re-categorization.

    Walks assistant messages, extracts tool_calls[i].id and the 'command' key from
    its JSON-encoded arguments. Used to re-classify role:tool messages based on
    the command that produced them.
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


def build_tool_call_url_map(messages: list) -> dict:
    """Map tool_call_id -> the URL a web_fetch-like call targeted.

    Looks for a `url`/`URL`/`uri`/`link` key in the JSON-decoded arguments
    first; falls back to regex-extracting the first http(s) URL from the raw
    arguments string for tools that nest the URL differently.
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
            except (json.JSONDecodeError, TypeError, ValueError):
                args = {}
            url = None
            if isinstance(args, dict):
                for key in ("url", "URL", "uri", "link"):
                    if args.get(key):
                        url = str(args[key])
                        break
            if url is None:
                args_str = args_raw if isinstance(args_raw, str) else json.dumps(args_raw)
                m = _URL_RE.search(args_str)
                if m:
                    url = m.group(0)
            if url:
                mapping[tc_id] = url
    return mapping


def build_tool_call_args_map(messages: list) -> dict:
    """Map tool_call_id -> parsed arguments dict.

    Third companion to build_tool_call_command_map / _fn_name_map: structured
    tools (Hermes `read_file`/`patch`/`write_file`/`execute_code`) carry their
    file identity in `path`-like or `code` args, NOT in a bash `command`, so
    file extraction for their tool-RESULT turns must go through the parsed
    args. Unparseable / non-dict arguments map to {}.
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
            mapping[tc_id] = args if isinstance(args, dict) else {}
    return mapping


# Structured-tool argument keys that name a file directly (taken verbatim as a
# path) vs. keys carrying a code payload (scanned with _FILE_PATH_RE, so a
# heredoc-style script's file references join the turn's file set). `content`
# / `text` args are deliberately NOT scanned: file BODIES routinely mention
# many unrelated paths and would over-link supersession.
_PATH_ARG_KEYS = ("path", "file_path", "filename", "file", "target_file")
_CODE_ARG_KEYS = ("code",)


def extract_files_from_tool_args(args: dict) -> set[str]:
    """File paths referenced by a structured tool call's parsed arguments."""
    files: set[str] = set()
    if not isinstance(args, dict):
        return files
    for key in _PATH_ARG_KEYS:
        v = args.get(key)
        if isinstance(v, str) and v.strip():
            files.add(v.strip())
    for key in _CODE_ARG_KEYS:
        v = args.get(key)
        if isinstance(v, str) and v:
            files.update(_FILE_PATH_RE.findall(v))
    return files


# Structured-tool function-name -> intent. Checked in EDIT, SEARCH, READ order so
# e.g. "rewrite"/"str_replace" (contain "write"/"replace") classify as edit, not
# read, and "file_search" classifies as search. Names vary across agents, so both
# exact tokens and substrings are matched. Bash/exec tools are intentionally
# absent: they carry a `command` and go through classify_bash_command.
_STRUCT_EDIT_KEYS = ("str_replace", "replace", "write", "edit", "patch", "insert",
                     "create_file", "createfile", "modify", "apply_diff", "apply_patch")
_STRUCT_SEARCH_KEYS = ("grep", "ripgrep", "glob", "search", "find", "locate",
                       "list_dir", "list_files", "listdir", "tree")
_STRUCT_READ_KEYS = ("read", "view", "cat", "open_file", "openfile", "show_file",
                     "get_file", "getfile", "fileread", "read_file")
# Code-execution tools that carry a `code` arg (not a `command`), so they miss
# the bash path entirely. Hermes `execute_code` is the canonical case. Checked
# LAST so file tools whose names happen to contain "run"/"exec" still win.
# "process" covers process-status tools (Hermes `process`): they report on a
# running exec, so their output belongs in the run/exec bucket, not TOOL_OTHER.
_STRUCT_EXEC_KEYS = ("execute_code", "run_code", "code_exec", "code_interpreter",
                     "python_exec", "ipython", "jupyter", "execute", "run_python",
                     "process")


def classify_structured_tool(fn_name: str) -> str:
    """Categorize a structured (non-bash) tool call by its function name.

    Returns one of TOOL_FILE_{EDIT,SEARCH,READ}, TOOL_RUN_EXEC, or ""
    when the name is not a recognized tool (caller then falls back to
    TOOL_OTHER).
    """
    n = (fn_name or "").strip().lower()
    if not n:
        return ""
    if any(k in n for k in _STRUCT_WEB_SEARCH_KEYS):
        return TOOL_WEB_SEARCH
    if any(k in n for k in _STRUCT_WEB_FETCH_KEYS):
        return TOOL_WEB_FETCH
    if any(k in n for k in _STRUCT_EDIT_KEYS):
        return TOOL_FILE_EDIT
    if any(k in n for k in _STRUCT_SEARCH_KEYS):
        return TOOL_FILE_SEARCH
    if any(k in n for k in _STRUCT_READ_KEYS):
        return TOOL_FILE_READ
    if any(k in n for k in _STRUCT_EXEC_KEYS):
        return TOOL_RUN_EXEC
    return ""


# Test-by-filename heuristic: `python3 test_foo.py`, `./tests/run.sh`,
# `python /app/test/test_environ.py` are test runs even though no recognized
# test-framework head appears. Matches a basename of test_*.* / *_test.* or a
# path component test/ | tests/. Applied only in the EXEC branches of
# classify_bash_command (python / direct-binary / bash), so reads like
# `cat test_foo.py` are unaffected.
_TEST_FILE_RE = re.compile(
    r"(?:^|/)(?:test_[\w.\-]+|[\w.\-]+_test)\.\w+$"
    r"|(?:^|/)tests?/"
)


def _mentions_test_file(tokens: list) -> bool:
    return any(
        _TEST_FILE_RE.search(t) for t in tokens
        if isinstance(t, str) and not t.startswith("-")
    )


_FILE_READ_HEADS = {"cat", "head", "tail", "less", "more", "view", "nl", "tac"}
_FILE_SEARCH_HEADS = {"find", "grep", "rg", "ag", "ack", "ls", "which",
                      "locate", "whereis", "tree"}
_RUN_EXEC_HEADS = {"node", "ruby", "perl", "lua", "java", "deno", "bun"}
_TEST_HEADS = {"pytest", "py.test", "tox", "nosetests", "jest", "mocha"}
_BUILD_HEADS = {"cmake", "ninja", "bazel", "buck", "mvn", "gradle"}
_STRUCT_WEB_SEARCH_KEYS = ("web_search", "websearch", "search_web", "google_search",
                           "brave_search", "bing_search", "internet_search")
_STRUCT_WEB_FETCH_KEYS = ("web_fetch", "webfetch", "fetch_url", "fetch_page",
                          "url_fetch", "browse", "visit_url", "read_url", "get_url")

def _strip_envs(tokens: list) -> list:
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


# Prefix commands that WRAP the real intent in the same chunk (`timeout 10
# python3 app.py`, `nohup python3 … &`, `exec python3 …`, `env FOO=1 make`).
# _first_head strips them (plus their own flags / duration args) and
# re-inspects what follows, so the wrapped command classifies by its intent
# instead of falling through to TOOL_OTHER.
_WRAPPER_HEADS = {"timeout", "nohup", "time", "exec", "setsid", "stdbuf",
                  "env", "unbuffer"}

# Duration-like token consumed by `timeout` (e.g. `timeout 10`, `timeout 5.5s`).
_DURATION_TOKEN_RE = re.compile(r"\d+(?:\.\d+)?[smhd]?$")


def _unwrap_wrappers(tokens: list) -> list:
    """Strip leading wrapper commands (+ their flags/args) from a token list."""
    toks = tokens
    while toks:
        toks = _strip_envs(toks)
        if not toks or toks[0] not in _WRAPPER_HEADS:
            break
        head = toks[0]
        toks = toks[1:]
        # The wrapper's own flags (`timeout -k 5`, `stdbuf -oL`, `env -i`).
        while toks and toks[0].startswith("-"):
            toks = toks[1:]
        if head == "timeout":
            # Drop the duration argument(s) (`timeout 10`, `timeout -k 5 10`).
            while toks and _DURATION_TOKEN_RE.fullmatch(toks[0]):
                toks = toks[1:]
    return toks


def _first_head(command: str) -> tuple:
    """Return (head, tokens_after_head) of the first sub-command in a chained line.

    Splits on shell operators (&&, ||, ;, |) and returns the head token of the
    first non-trivial sub-command (skipping `cd` prefixes which are navigation,
    not the intent). Strips env-var assignments and wrapper prefixes
    (timeout/nohup/exec/…). The returned head is the *intent* of the line for
    classification purposes.
    """
    if not command:
        return ("", [])
    # Split on shell operators (very rough but good enough for classification)
    chunks = re.split(r"(?:&&|\|\||;|\|)", command)
    for raw in chunks:
        toks = raw.strip().split()
        if not toks:
            continue
        toks = _strip_envs(toks)
        if not toks:
            continue
        head = toks[0]
        # Skip pure navigation / no-op prefixes (cd, pwd, env-setters, etc.).
        # These are scaffolding; the real intent of the line is the next chunk.
        # `eval` is here for its dominant scaffold form `eval $(opam env) && …`
        # (an eval'd literal command still classifies via the fallback below);
        # `sleep`/`wait` are here for retry/probe scaffolds like
        # `sleep 3 && curl …`.
        if head in (
            "cd", "pushd", "popd", "pwd", "set", "export", "unset", "true",
            ":", "source", ".", "umask", "alias", "ulimit",
            "eval", "sleep", "wait",
        ):
            continue
        toks = _unwrap_wrappers(toks)
        if not toks:
            continue
        return (toks[0], toks[1:])
    # All chunks were skips -- fall back to the very first token
    toks = command.strip().split()
    toks = _strip_envs(toks)
    if toks:
        return (toks[0], toks[1:])
    return ("", [])


def classify_bash_command(command: str, content: str) -> str:
    """Classify a bash command into one of the 7 tool-output sub-buckets.

    See module-level comment in scripts/analyze_tokens.py for heuristic priority
    and edge cases. `content` is accepted for symmetry with
    `_classify_bash_observation` and to allow future content-aware refinements
    (e.g. detecting `>500` chars edit echoes as reads), but the v1
    implementation classifies purely on the command.
    """
    head, rest = _first_head((command or "").strip())
    if not head:
        return TOOL_OTHER

    # 1. File search (must come BEFORE file-read so `grep -r` doesn't get
    #    miscategorized as a read of the search-pattern argument).
    if head in _FILE_SEARCH_HEADS:
        return TOOL_FILE_SEARCH

    # 2. File read
    if head in _FILE_READ_HEADS:
        return TOOL_FILE_READ
    if head == "sed" and rest and rest[0].startswith("-n"):
        return TOOL_FILE_READ

    # 3. Test runners
    if head in _TEST_HEADS:
        return TOOL_TEST_RUN
    if head in ("python", "python3", "python2"):
        # `python -m pytest`, `python -m unittest`, `python -m nose`
        if len(rest) >= 2 and rest[0] == "-m" and rest[1] in (
            "pytest", "unittest", "nose", "nose2", "tox"
        ):
            return TOOL_TEST_RUN
        # `python -m pip install` -> build/install
        if len(rest) >= 3 and rest[0] == "-m" and rest[1] == "pip" and "install" in rest[2:]:
            return TOOL_BUILD_INSTALL
        # `python test_foo.py` / `python tests/…` -> test run by filename
        if _mentions_test_file(rest):
            return TOOL_TEST_RUN
        # everything else python -> run/exec
        return TOOL_RUN_EXEC

    # 4. Build / install
    if head in _BUILD_HEADS:
        return TOOL_BUILD_INSTALL
    if head == "pip" and rest and rest[0] in ("install", "uninstall", "wheel"):
        return TOOL_BUILD_INSTALL
    if head in ("conda", "mamba", "micromamba") and rest and rest[0] in ("install", "create", "update"):
        return TOOL_BUILD_INSTALL
    if head in ("apt", "apt-get", "yum", "dnf", "pacman", "brew") and rest and rest[0] in ("install", "update", "upgrade"):
        return TOOL_BUILD_INSTALL
    if head == "make":
        # `make test`/`make check` -> test_run; everything else (incl. bare `make`,
        # `make install`) is build_install.
        if rest and rest[0] in ("test", "tests", "check", "checks"):
            return TOOL_TEST_RUN
        return TOOL_BUILD_INSTALL
    if head in ("npm", "yarn", "pnpm"):
        if rest and rest[0] in ("test", "run") and len(rest) >= 2 and rest[1] in ("test", "tests"):
            return TOOL_TEST_RUN
        if rest and rest[0] == "test":
            return TOOL_TEST_RUN
        if rest and rest[0] in ("install", "i", "add", "ci"):
            return TOOL_BUILD_INSTALL
        return TOOL_RUN_EXEC
    if head == "cargo":
        if rest and rest[0] in ("test", "bench"):
            return TOOL_TEST_RUN
        if rest and rest[0] in ("build", "install"):
            return TOOL_BUILD_INSTALL
        if rest and rest[0] == "run":
            return TOOL_RUN_EXEC
        return TOOL_BUILD_INSTALL
    if head == "go":
        if rest and rest[0] == "test":
            return TOOL_TEST_RUN
        if rest and rest[0] in ("build", "install", "get", "mod"):
            return TOOL_BUILD_INSTALL
        if rest and rest[0] == "run":
            return TOOL_RUN_EXEC

    # 5. File edit (after build to avoid `make install > log` matching `>`)
    if head == "sed" and rest and any(r.startswith("-i") for r in rest):
        return TOOL_FILE_EDIT
    if head in ("patch", "tee"):
        return TOOL_FILE_EDIT
    if head in ("mv", "cp", "rm", "mkdir", "touch", "ln", "chmod", "chown"):
        return TOOL_FILE_EDIT
    # echo/printf into a file via shell redirect -- head is echo/printf and the
    # full command contains `>` or `>>`
    if head in ("echo", "printf") and (" > " in command or " >> " in command):
        return TOOL_FILE_EDIT
    # heredoc patterns: `cat <<EOF > file` writes a file
    if head == "cat" and ("<<" in command and (" > " in command or " >> " in command)):
        return TOOL_FILE_EDIT

    # 6. Run / exec (interpreters not caught above)
    if head in _RUN_EXEC_HEADS:
        if _mentions_test_file([head] + rest):
            return TOOL_TEST_RUN
        return TOOL_RUN_EXEC
    if head.startswith("./") or head.startswith("/"):
        # Direct binary execution
        if _mentions_test_file([head] + rest):
            return TOOL_TEST_RUN
        return TOOL_RUN_EXEC
    if head in ("bash", "sh", "zsh", "ksh"):
        if _mentions_test_file(rest):
            return TOOL_TEST_RUN
        return TOOL_RUN_EXEC

    # 7. Default
    return TOOL_OTHER


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


# Observation categories (tool outputs)
OBSERVATION_CATEGORIES = {
    TOOL_FILE_READ, TOOL_FILE_EDIT, TOOL_FILE_SEARCH,
    TOOL_WEB_FETCH, TOOL_WEB_SEARCH,
    TOOL_TEST_RUN, TOOL_BUILD_INSTALL, TOOL_RUN_EXEC,
    TOOL_OTHER,
}

# Categories eligible for content-hash dedupe of REPEATED command output
# (findings doc §5 primary recommendation). Path-based supersession
# (superseded_by_later_read / _by_edit) already covers file reads/edits; the
# un-deduped leak is re-running the same script / test / `ls` whose output is
# near-identical but is NOT file content, so path overlap never fires. Restricted
# to run/exec + other-bash: test/build runs are intentionally LEFT OUT because a
# rerun whose output is byte-identical-after-normalization is exactly what we
# would WANT to keep distinguishable for the agent's reasoning; run/exec +
# other-bash are the high-volume, low-signal repeats (≈29-34% of tokens).
DEDUPE_OUTPUT_CATEGORIES = {TOOL_RUN_EXEC, TOOL_OTHER}

# Categories participating in run-target supersession (supersede_reruns): the
# outputs of PROGRAM EXECUTION, whose relevance is tied to the version of the
# code they ran -- a later run (or edit; that edge rides supersede_reads)
# touching the same target file makes the earlier output stale. TEST_RUN is in
# the set but guarded: a test output may only be superseded by a later
# SUCCESSFUL test run of the same target, preserving the pass/fail trajectory.
RUN_OUTPUT_CATEGORIES = {
    TOOL_RUN_EXEC, TOOL_OTHER, TOOL_BUILD_INSTALL, TOOL_TEST_RUN,
}

# Paths never treated as run TARGETS for supersession: the exec bridge's own
# snapshot/scratch files live under /tmp and would join every command into one
# giant equivalence class.
_RUN_TARGET_EXCLUDE_PREFIXES = ("/tmp/", "/var/folders/")


def run_target_files(files_full: set[str]) -> set[str]:
    """Filter a turn's full-path file set down to run-target candidates."""
    return {
        f for f in files_full
        if f and not f.startswith(_RUN_TARGET_EXCLUDE_PREFIXES)
    }


def observation_is_failure(text: str) -> bool:
    """True iff a tool observation carries a nonzero exit code.

    Hermes-via-exec-bridge wraps tool results as JSON
    ``{"output": …, "exit_code": N, "error": …}`` so the exit code is
    recoverable server-side from the message content alone. Unknown shapes
    (plain-text output, other harnesses) return False -- i.e. treat as
    success, the conservative branch for supersession policies.
    """
    if not text:
        return False
    s = text.lstrip()
    if not s.startswith("{"):
        return False
    try:
        obj = json.loads(s)
    except Exception:
        return False
    if not isinstance(obj, dict):
        return False
    ec = obj.get("exit_code")
    return isinstance(ec, int) and ec != 0

# Minimum normalized-text length before a run/exec / other-bash observation is
# considered for dedupe. Tiny outputs (empty, a single prompt line, "OK") carry
# negligible tokens and have a higher chance of coincidental collision, so we
# skip them rather than risk evicting unrelated short outputs.
_MIN_DEDUPE_CHARS = 40

# Volatile-token scrubbers for content-hash dedupe. Each pattern replaces a class
# of run-to-run-varying noise with a fixed placeholder so re-running the SAME
# command (whose output differs only in timing / addresses / tmp names) collapses
# to one normalized form. Patterns are deliberately NARROW: over-scrubbing would
# collapse genuinely-different outputs and evict context the agent still needs.
_NORM_PATTERNS = [
    # ISO-8601 / log timestamps: 2026-06-24T12:34:56(.123)(Z|+08:00),
    # 2026-06-24 12:34:56
    (re.compile(
        r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}"
        r"(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?"), "<TS>"),
    # bare wall-clock times HH:MM:SS(.frac)
    (re.compile(r"\b\d{2}:\d{2}:\d{2}(?:[.,]\d+)?\b"), "<TS>"),
    # elapsed durations: "0.53s", "1.2 ms", "in 3.45 seconds", "12.3ms"
    (re.compile(
        r"\b\d+(?:\.\d+)?\s*"
        r"(?:ms|s|sec|secs|seconds|min|mins|minutes|us|µs|ns)\b"), "<DUR>"),
    # hex addresses / object ids: 0xdeadbeef, "at 0x7f1234"
    (re.compile(r"\b0x[0-9a-fA-F]+\b"), "<ADDR>"),
    # process-specific temp paths (pytest tmpdirs, mktemp): /tmp/..., /var/folders/...
    (re.compile(r"/tmp/[\w./\-]+"), "<TMP>"),
    (re.compile(r"/var/folders/[\w./\-]+"), "<TMP>"),
]


def normalize_observation_text(text: str) -> str:
    """Normalize a command-output observation for repeat detection.

    Strips the volatile noise (timestamps, elapsed durations, hex addresses,
    process-temp paths) that makes two runs of the SAME command differ, then
    collapses all whitespace. Two outputs that normalize to the same string are
    treated as a repeat by the tracker's content-hash dedupe. Returns "" for
    empty / whitespace-only input.
    """
    if not text:
        return ""
    out = text
    for pat, repl in _NORM_PATTERNS:
        out = pat.sub(repl, out)
    return re.sub(r"\s+", " ", out).strip()


def _classify_new_messages(new_msgs: list[dict], all_msgs_so_far: list[dict]) -> tuple[str, str]:
    """Determine the primary category of new messages and extract the bash command.

    Returns (category, command).
    """
    if not new_msgs:
        return AGENT_REASONING, ""

    # Check what roles are present
    roles = [m.get("role") for m in new_msgs]

    # If system message present, it's the system prompt turn
    if "system" in roles:
        return SYSTEM_PROMPT, ""

    # If user message present (and no assistant), it's user task
    if "user" in roles and "assistant" not in roles:
        return USER_TASK, ""

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
                    category = classify_bash_command(command, content)
                    return category, command
                # Structured tool with no bash command (e.g. Hermes read/grep/edit
                # carrying a `path` arg): categorize by the tool function name so
                # file reads are recognized and can be superseded/evicted.
                struct = classify_structured_tool(
                    build_tool_call_fn_name_map(all_msgs).get(tc_id, ""))
                if struct:
                    return struct, ""
                return TOOL_OTHER, ""
        return TOOL_OTHER, ""

    # If assistant message with tool_calls, it's an agent tool call
    for msg in new_msgs:
        if msg.get("role") == "assistant":
            if msg.get("tool_calls"):
                return AGENT_TOOL_CALL, ""
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
                    return AGENT_TOOL_CALL, ""
            return AGENT_REASONING, ""

    return AGENT_REASONING, ""


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

    # Also extract file paths from tool messages: the bash command that
    # produced them AND the structured args of the originating tool call
    # (`path`/`code` for Hermes read_file/patch/write_file/execute_code --
    # without this, structured-tool RESULT turns carry no files and can
    # neither supersede nor be superseded).
    cmd_map = build_tool_call_command_map(all_msgs + new_msgs)
    args_map = build_tool_call_args_map(all_msgs + new_msgs)
    for msg in new_msgs:
        if msg.get("role") in ("tool", "function"):
            tc_id = msg.get("tool_call_id", "")
            command = cmd_map.get(tc_id, "")
            # Extract file paths from the command itself
            if command:
                for match in _FILE_PATH_RE.findall(command):
                    if match:
                        files.add(match)
            files.update(extract_files_from_tool_args(args_map.get(tc_id, {})))

    return files


def _extract_urls_from_messages(
    new_msgs: list[dict], all_msgs: list[dict], category: str,
) -> set[str]:
    """Extract normalized URLs referenced by a single web_search/web_fetch turn.

    The meaning of "referenced" depends on `category` -- mirrors how
    `files_referenced` doubles for both reads and edits, disambiguated by
    `is_edit` elsewhere:

    - TOOL_WEB_SEARCH: URLs found in the tool OUTPUT content (the candidate
      links the query surfaced -- what a later web_fetch might "consume").
    - TOOL_WEB_FETCH: the URL argument the call targeted (falls back to
      scanning the tool output content if no `url`-like argument was found).
    - anything else: empty set (this turn carries no web signal).
    """
    if category not in (TOOL_WEB_SEARCH, TOOL_WEB_FETCH):
        return set()

    urls: set[str] = set()
    all_together = all_msgs + new_msgs
    for msg in new_msgs:
        if msg.get("role") not in ("tool", "function"):
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in content
            )
        if category == TOOL_WEB_SEARCH:
            for raw in _URL_RE.findall(content or ""):
                urls.add(_normalize_url(raw))
        else:  # TOOL_WEB_FETCH
            tc_id = msg.get("tool_call_id", "")
            target = build_tool_call_url_map(all_together).get(tc_id)
            if target:
                urls.add(_normalize_url(target))
            else:
                for raw in _URL_RE.findall(content or ""):
                    urls.add(_normalize_url(raw))
                    break
    return urls


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
