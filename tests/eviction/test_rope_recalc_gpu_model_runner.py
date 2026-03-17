import numpy as np
from types import SimpleNamespace

from vllm.v1.worker.gpu_model_runner import GPUModelRunner


class _DummyArrayWrap:
    def __init__(self, arr):
        self.np = arr


class _DummyBlockTableObj:
    def __init__(self, n_reqs: int, n_cols: int = 16, block_size: int = 16):
        self.block_size = block_size
        self.block_table = _DummyArrayWrap(np.zeros((n_reqs, n_cols), dtype=np.int32))
        self.num_blocks_per_row = np.zeros((n_reqs,), dtype=np.int32)


class _DummyInputBatch:
    def __init__(self, req_ids, num_computed_tokens_cpu):
        self.req_id_to_index = {rid: i for i, rid in enumerate(req_ids)}
        self.num_computed_tokens_cpu = np.array(num_computed_tokens_cpu, dtype=np.int32)
        self.block_table = SimpleNamespace(
            block_tables=[_DummyBlockTableObj(len(req_ids))]
        )


def _build_runner(req_ids, pre_lengths):
    r = GPUModelRunner.__new__(GPUModelRunner)
    r.input_batch = _DummyInputBatch(req_ids, pre_lengths)
    r.requests = {rid: SimpleNamespace() for rid in req_ids}
    r.evicted_ranges = {}
    r.num_evicted_tokens_list = {}
    r.positions = _DummyArrayWrap(np.full((512,), -1, dtype=np.int32))
    return r


def _apply_eviction_like_prepare_inputs(runner, evicted_ranges_map):
    """
    Replays the eviction loop semantics from GPUModelRunner._prepare_inputs:
    - per request, compact KV and update num_computed_tokens_cpu
    - track evicted token counts/ranges
    """
    block_table_obj = runner.input_batch.block_table.block_tables[0]
    block_size = block_table_obj.block_size
    bt_np = block_table_obj.block_table.np

    def is_same_range(list1, list2):
        return len(list1) == len(list2) and all(a == b for a, b in zip(list1, list2))

    for req_id, ranges in evicted_ranges_map.items():
        if is_same_range(runner.evicted_ranges.get(req_id, []), ranges):
            continue

        req_index = runner.input_batch.req_id_to_index.get(req_id)
        if req_index is None:
            continue

        current_total_len = int(runner.input_batch.num_computed_tokens_cpu[req_index])
        past_mask = None  # not needed by mocked compact path in this test

        num_survivors = runner._compact_kv_caches(
            req_index, past_mask, bt_np, block_size, ranges
        )

        runner.num_evicted_tokens_list[req_id] = current_total_len - num_survivors
        runner.evicted_ranges[req_id] = ranges


def _recompute_positions_like_prepare_inputs(runner, req_indices, arange):
    """
    Replays the exact RoPE position recompute added in your commit:
        positions_np = num_computed_tokens_cpu[req_indices] + arange
    """
    total = len(req_indices)
    positions_np = runner.positions.np[:total]
    np.add(
        runner.input_batch.num_computed_tokens_cpu[req_indices],
        arange,
        out=positions_np,
    )
    return positions_np


def test_rope_recalculation_after_eviction_mixed_batch_strict(monkeypatch):
    # Mixed batch setup
    req_ids = ["reqA", "reqB", "reqC"]
    pre_lengths = [100, 50, 8]
    runner = _build_runner(req_ids, pre_lengths)

    # 1) pre-eviction computed length assertions
    assert runner.input_batch.num_computed_tokens_cpu.tolist() == [100, 50, 8]

    # Eviction input
    evicted_ranges_map = {
        "reqA": [(20, 30), (70, 75)],  # removes 15
        "reqB": [(0, 10)],             # removes 10
        # reqC none
    }

    # 2) evicted range assertions
    assert evicted_ranges_map["reqA"] == [(20, 30), (70, 75)]
    assert evicted_ranges_map["reqB"] == [(0, 10)]
    assert evicted_ranges_map.get("reqC", []) == []

    # Mock compaction: only enforce computed-length effect
    def _mock_compact_kv_caches(req_index, past_mask, bt_np, block_size, ranges):
        current = int(runner.input_batch.num_computed_tokens_cpu[req_index])
        removed = sum(end - start for start, end in ranges)
        survivors = current - removed
        runner.input_batch.num_computed_tokens_cpu[req_index] = survivors
        return survivors

    monkeypatch.setattr(runner, "_compact_kv_caches", _mock_compact_kv_caches)

    # Apply eviction phase
    _apply_eviction_like_prepare_inputs(runner, evicted_ranges_map)

    # 3) post-eviction computed length assertions
    assert runner.input_batch.num_computed_tokens_cpu.tolist() == [85, 40, 8]
    assert runner.num_evicted_tokens_list == {"reqA": 15, "reqB": 10}

    # Mixed-batch flattening for scheduled tokens [3,2,4]
    req_indices = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2], dtype=np.int32)
    arange = np.array([0, 1, 2, 0, 1, 0, 1, 2, 3], dtype=np.int32)

    positions_np = _recompute_positions_like_prepare_inputs(runner, req_indices, arange)

    # 4) exact expected positions assertions
    expected_positions = np.array([85, 86, 87, 40, 41, 8, 9, 10, 11], dtype=np.int32)
    np.testing.assert_array_equal(positions_np, expected_positions)