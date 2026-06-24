"""Per-observation-message token-sequence hash index (D-07 third view, D-11/D-12).

Hash algorithm: `hashlib.blake2b(digest_size=8)`. RESEARCH Finding 6 documents why
xxhash is intentionally NOT used: it is an OPTIONAL vLLM dep
(`requirements/common.txt` does not include it; only `requirements/test.txt` does).
Importing xxhash unconditionally would crash the vLLM server at startup on minimal
installs. blake2b is stdlib, deterministic across Python versions, and produces a
same-width (8-byte) digest. The discretion line in CONTEXT D-discussion permits this.

`compute_message_token_ranges` (the per-message token-range derivation, RESEARCH
Finding 7) is NOT defined here -- it lives in `tracker.py` because it depends on
the chat-template + tokenizer arguments that the serving hook hands in.
"""
from __future__ import annotations

import hashlib
import struct


def hash_token_sequence(token_ids: list[int]) -> bytes:
    """64-bit blake2b digest of a token-id sequence (D-11/D-12, RESEARCH Finding 6).

    Used as TokenSequenceIndex key. The digest is 8 bytes, same width as xxhash64,
    but uses stdlib hashlib so the tracker has no optional-dep gotcha.

    Empty sequences are hashed too -- the digest of `b""` is well-defined and
    callers should NOT need to special-case empty observations.
    """
    h = hashlib.blake2b(digest_size=8)
    if token_ids:
        # struct-pack int32 each: predictable byte width, faster than bytes(token_ids)
        # because token ids can exceed 255 (Qwen vocab is ~150k).
        h.update(struct.pack(f"<{len(token_ids)}i", *token_ids))
    return h.digest()


def hash_text(text: str) -> bytes:
    """64-bit blake2b digest of a (already-normalized) text string.

    Companion to hash_token_sequence for content-hash dedupe of repeated
    command output (findings doc §5): the dedupe keys on NORMALIZED observation
    TEXT (timestamps/paths/whitespace scrubbed) rather than raw token ids, so
    near-identical reruns collapse to one digest. Same stdlib-only, 8-byte,
    cross-version-deterministic guarantees as hash_token_sequence.
    """
    h = hashlib.blake2b(digest_size=8)
    if text:
        h.update(text.encode("utf-8"))
    return h.digest()


class TokenSequenceIndex:
    """`hash_token_sequence(observation_tokens) -> [turn_idx]` (D-07).

    Per-observation-message exact match only -- no sliding window, no chunk
    hashing (D-11; Phase 01.3 D-06 found near-misses are not viable).
    """

    def __init__(self) -> None:
        self._index: dict[bytes, list[int]] = {}

    def add(self, token_hash: bytes, turn_idx: int) -> None:
        self._index.setdefault(token_hash, []).append(turn_idx)

    def lookup(self, token_hash: bytes) -> list[int]:
        return list(self._index.get(token_hash, ()))

    def exact_matches(self) -> list[tuple[int, int]]:
        """Return list of `(later_turn_idx, earliest_match_turn_idx)` pairs.

        Used to populate `OpportunityResponse.exact_match_turns` (D-16).
        For each hash with >=2 turns, every entry after the first is a match.
        """
        pairs: list[tuple[int, int]] = []
        for turns in self._index.values():
            if len(turns) >= 2:
                earliest = turns[0]
                for later in turns[1:]:
                    pairs.append((later, earliest))
        return pairs
