"""UrlTimeline data structure (D-20): compact record of which URLs a session
has already searched or fetched.

Mirrors FileTimeline (file_timeline.py) but keyed by normalized URL instead
of file basename. This is the "one thing worth keeping a trace of" from the
web-search eviction rule: once a web_search turn's raw result snippet is
evicted, this index is what still lets the opportunity map show that a URL
was already surfaced/fetched, without retaining the snippet content itself.
"""
from __future__ import annotations

from typing import NamedTuple


class UrlTimelineEntry(NamedTuple):
    turn_idx: int
    action: str      # "search" | "fetch"
    msg_idx: int


class UrlTimeline:
    """`normalized_url -> list[UrlTimelineEntry]` (D-20, ordered by turn_idx)."""

    def __init__(self) -> None:
        self._by_url: dict[str, list[UrlTimelineEntry]] = {}

    def append(self, *, url: str, turn_idx: int, action: str, msg_idx: int) -> None:
        self._by_url.setdefault(url, []).append(
            UrlTimelineEntry(turn_idx=turn_idx, action=action, msg_idx=msg_idx)
        )

    def get(self, url: str) -> list[UrlTimelineEntry]:
        return self._by_url.get(url, [])

    def urls(self) -> list[str]:
        return list(self._by_url.keys())

    def to_dict(self) -> dict[str, list[dict]]:
        """Serializable shape for the GET /opportunity response (D-16)."""
        return {
            u: [{"turn_idx": e.turn_idx, "action": e.action, "msg_idx": e.msg_idx}
                for e in entries]
            for u, entries in self._by_url.items()
        }
