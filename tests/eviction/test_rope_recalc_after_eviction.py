import numpy as np


def apply_eviction_and_recompute_positions(
    num_computed_tokens_cpu: np.ndarray,
    req_ids,
    evictable_token_ranges_map,
    num_scheduled_tokens,
):
    """
    Minimal, deterministic model of the logic we need to validate:
      1) update per-request computed length after eviction
      2) recompute positions_np via:
            positions_np = num_computed_tokens_cpu[req_indices] + arange
    """

    # Pre-eviction snapshot for assertions.
    pre = {rid: int(num_computed_tokens_cpu[idx]) for idx, rid in enumerate(req_ids)}

    # Apply eviction effect to per-request computed length.
    # Assumes half-open token ranges [start, end), matching common scheduler semantics.
    for req_index, req_id in enumerate(req_ids):
        ranges = evictable_token_ranges_map.get(req_id, [])
        removed = 0
        for start, end in ranges:
            assert 0 <= start <= end <= pre[req_id], (
                f"Invalid eviction range {(start, end)} for request {req_id} "
                f"with pre length {pre[req_id]}"
            )
            removed += end - start
        num_computed_tokens_cpu[req_index] -= removed

    post = {rid: int(num_computed_tokens_cpu[idx]) for idx, rid in enumerate(req_ids)}

    # Build flattened req_indices exactly like mixed-batch scheduling would.
    req_indices = np.concatenate([
        np.full(n, i, dtype=np.int32) for i, n in enumerate(num_scheduled_tokens)
    ])
    arange = np.concatenate([
        np.arange(n, dtype=np.int32) for n in num_scheduled_tokens
    ])

    positions_np = np.empty_like(req_indices, dtype=np.int32)
    np.add(num_computed_tokens_cpu[req_indices], arange, out=positions_np)

    return pre, post, req_indices, arange, positions_np


def test_rope_recalculation_after_eviction_mixed_batch():
    # Mixed batch: 3 requests, different computed lengths and scheduled token counts.
    req_ids = ["reqA", "reqB", "reqC"]
    num_computed_tokens_cpu = np.array([100, 50, 8], dtype=np.int32)
    num_scheduled_tokens = [3, 2, 4]  # reqA gets 3, reqB gets 2, reqC gets 4

    # Evict ranges for only some requests; reqC has no eviction.
    evictable_token_ranges_map = {
        "reqA": [(20, 30), (70, 75)],  # remove 10 + 5 = 15
        "reqB": [(0, 10)],             # remove 10
        # reqC: no eviction
    }

    pre, post, req_indices, arange, positions_np = apply_eviction_and_recompute_positions(
        num_computed_tokens_cpu=num_computed_tokens_cpu.copy(),
        req_ids=req_ids,
        evictable_token_ranges_map=evictable_token_ranges_map,
        num_scheduled_tokens=num_scheduled_tokens,
    )

    # 1) Assert pre-eviction computed length.
    assert pre == {"reqA": 100, "reqB": 50, "reqC": 8}

    # 2) Assert evicted ranges are exactly what we intended to test.
    assert evictable_token_ranges_map["reqA"] == [(20, 30), (70, 75)]
    assert evictable_token_ranges_map["reqB"] == [(0, 10)]
    assert evictable_token_ranges_map.get("reqC", []) == []

    # 3) Assert post-eviction computed length.
    assert post == {"reqA": 85, "reqB": 40, "reqC": 8}

    # 4) Assert exact expected positions_np in mixed-batch flattening order.
    # Flatten order from num_scheduled_tokens [3,2,4]:
    # req_indices = [A,A,A, B,B, C,C,C,C]
    # arange      = [0,1,2, 0,1, 0,1,2,3]
    # positions   = [85,86,87, 40,41, 8,9,10,11]
    expected_req_indices = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2], dtype=np.int32)
    expected_arange = np.array([0, 1, 2, 0, 1, 0, 1, 2, 3], dtype=np.int32)
    expected_positions = np.array([85, 86, 87, 40, 41, 8, 9, 10, 11], dtype=np.int32)

    np.testing.assert_array_equal(req_indices, expected_req_indices)
    np.testing.assert_array_equal(arange, expected_arange)
    np.testing.assert_array_equal(positions_np, expected_positions)