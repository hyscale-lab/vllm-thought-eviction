"""FileTimeline data structure (D-07, first indexed view).

`file_basename -> [FileTimelineEntry]`. Drives supersession queries
(Phase 01.3 D-02 file-path overlap detection).
"""
from __future__ import annotations

from typing import NamedTuple


class FileTimelineEntry(NamedTuple):
    turn_idx: int
    action: str         # "read" | "edit" | "search" | "test" | "build" | "other"
    msg_idx: int
    full_path: str


class FileTimeline:
    """`file_basename -> list[FileTimelineEntry]` (D-07, ordered by turn_idx)."""

    def __init__(self) -> None:
        self._by_basename: dict[str, list[FileTimelineEntry]] = {}

    def append(self, *, basename: str, turn_idx: int, action: str,
               msg_idx: int, full_path: str) -> None:
        self._by_basename.setdefault(basename, []).append(
            FileTimelineEntry(turn_idx=turn_idx, action=action,
                              msg_idx=msg_idx, full_path=full_path)
        )

    def get(self, basename: str) -> list[FileTimelineEntry]:
        return self._by_basename.get(basename, [])

    def basenames(self) -> list[str]:
        return list(self._by_basename.keys())

    def to_dict(self) -> dict[str, list[dict]]:
        """Serializable shape for the GET /opportunity response (D-16)."""
        return {
            bn: [{"turn_idx": e.turn_idx, "action": e.action,
                  "msg_idx": e.msg_idx, "full_path": e.full_path}
                 for e in entries]
            for bn, entries in self._by_basename.items()
        }
