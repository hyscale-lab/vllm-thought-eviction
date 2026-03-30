"""Tests for per-request IPC guard via SamplingParams.enable_l2_norms (Phase 6).

Covers:
- GUARD-01: Non-eviction requests produce zero L2 norm computation
- GUARD-02: SamplingParams.enable_l2_norms=False is omitted from serialization

Requirements: GUARD-01, GUARD-02
"""

import sys
import unittest

import msgspec
import torch


class TestSamplingParamsEnableL2Norms(unittest.TestCase):
    """SamplingParams.enable_l2_norms field — default, setter, serialization."""

    def test_default_false(self):
        """Field defaults to False so standard requests have zero overhead."""
        from vllm.sampling_params import SamplingParams
        sp = SamplingParams()
        self.assertIs(sp.enable_l2_norms, False)

    def test_set_true(self):
        """Field can be set to True for eviction-enabled requests."""
        from vllm.sampling_params import SamplingParams
        sp = SamplingParams(enable_l2_norms=True)
        self.assertIs(sp.enable_l2_norms, True)

    def test_omit_defaults_excludes_false(self):
        """GUARD-02: enable_l2_norms=False is omitted from msgpack serialization."""
        from vllm.sampling_params import SamplingParams
        encoded_default = msgspec.msgpack.encode(SamplingParams())
        encoded_true = msgspec.msgpack.encode(SamplingParams(enable_l2_norms=True))
        self.assertGreater(
            len(encoded_true), len(encoded_default),
            "enable_l2_norms=True should be larger than omitted default",
        )


class TestUpdateNormsBatchEvictionFilter(unittest.TestCase):
    """update_norms_batch eviction_request_ids filter — GUARD-01 inner loop."""

    def _reset_cache(self):
        from vllm.v1.attention.l2_norm_cache import L2NormCache
        L2NormCache._instance = None
        return L2NormCache()

    def _make_tensors(self, num_requests: int = 1, seq_len: int = 4,
                      block_size: int = 4, num_blocks: int = 2):
        """Build minimal mock tensors for update_norms_batch."""
        # key_cache: one layer, shape [num_blocks, block_size, num_heads, head_size]
        key_cache = [torch.randn(num_blocks, block_size, 2, 8)]
        # block_table: one entry per layer, shape [num_requests, max_blocks_per_req]
        block_table = [torch.zeros(num_requests, 1, dtype=torch.int32)]
        for req_idx in range(num_requests):
            block_table[0][req_idx, 0] = req_idx  # each request owns its own block
        # seq_lens: one entry per request
        seq_lens = torch.full((num_requests,), seq_len, dtype=torch.int32)
        return key_cache, block_table, seq_lens

    def test_skips_non_eviction_request(self):
        """Non-eviction request in empty eviction_request_ids set produces no norms."""
        cache = self._reset_cache()
        cache.set_request_layers("req-noevict", None)
        key_cache, block_table, seq_lens = self._make_tensors(num_requests=1)
        cache.update_norms_batch(
            request_ids=["req-noevict"],
            key_cache=key_cache,
            block_table=block_table,
            seq_lens=seq_lens,
            block_size=4,
            eviction_request_ids=set(),
        )
        self.assertIsNone(
            cache.get_norms("req-noevict"),
            "Non-eviction request should have no norms when excluded from eviction_request_ids",
        )

    def test_processes_eviction_request(self):
        """Eviction-enabled request in eviction_request_ids receives computed norms."""
        cache = self._reset_cache()
        cache.set_request_layers("req-evict", None)
        key_cache, block_table, seq_lens = self._make_tensors(num_requests=1)
        cache.update_norms_batch(
            request_ids=["req-evict"],
            key_cache=key_cache,
            block_table=block_table,
            seq_lens=seq_lens,
            block_size=4,
            eviction_request_ids={"req-evict"},
        )
        norms = cache.get_norms("req-evict")
        self.assertIsNotNone(norms, "Eviction request should have norms")
        self.assertGreater(len(norms), 0, "Norms list should be non-empty")

    def test_mixed_batch_only_processes_eviction(self):
        """In a mixed batch, only the eviction-enabled request receives norms."""
        cache = self._reset_cache()
        cache.set_request_layers("req-evict", None)
        cache.set_request_layers("req-std", None)
        key_cache, block_table, seq_lens = self._make_tensors(num_requests=2)
        cache.update_norms_batch(
            request_ids=["req-evict", "req-std"],
            key_cache=key_cache,
            block_table=block_table,
            seq_lens=seq_lens,
            block_size=4,
            eviction_request_ids={"req-evict"},
        )
        evict_norms = cache.get_norms("req-evict")
        self.assertIsNotNone(evict_norms, "Eviction request should have norms")
        self.assertGreater(len(evict_norms), 0)
        self.assertIsNone(
            cache.get_norms("req-std"),
            "Standard request should have no norms when excluded from eviction_request_ids",
        )

    def test_none_eviction_ids_processes_all(self):
        """eviction_request_ids=None disables filtering — all active requests are processed."""
        cache = self._reset_cache()
        cache.set_request_layers("req-1", None)
        key_cache, block_table, seq_lens = self._make_tensors(num_requests=1)
        cache.update_norms_batch(
            request_ids=["req-1"],
            key_cache=key_cache,
            block_table=block_table,
            seq_lens=seq_lens,
            block_size=4,
            eviction_request_ids=None,
        )
        norms = cache.get_norms("req-1")
        self.assertIsNotNone(norms, "Request should have norms when eviction_request_ids=None")
        self.assertGreater(len(norms), 0)


class TestSchedulerGuardSource(unittest.TestCase):
    """Scheduler uses sampling_params.enable_l2_norms guard (source inspection)."""

    def test_scheduler_uses_sampling_params_guard(self):
        """Scheduler norm fetch is gated on request.sampling_params.enable_l2_norms."""
        import pathlib
        src = pathlib.Path(
            "/export/home2/broc/vllm-thought-eviction/vllm/v1/core/sched/scheduler.py"
        ).read_text()
        self.assertIn(
            "sampling_params.enable_l2_norms",
            src,
            "Scheduler must guard norm fetch with sampling_params.enable_l2_norms",
        )

    def test_scheduler_no_singleton_guard(self):
        """Scheduler must not use the old cross-process is_eviction_active singleton guard."""
        import pathlib
        src = pathlib.Path(
            "/export/home2/broc/vllm-thought-eviction/vllm/v1/core/sched/scheduler.py"
        ).read_text()
        self.assertNotIn(
            "is_eviction_active",
            src,
            "Scheduler must not use is_eviction_active (cross-process singleton anti-pattern)",
        )


class TestGpuModelRunnerGuardSource(unittest.TestCase):
    """gpu_model_runner uses enable_l2_norms at all three guard sites (source inspection)."""

    def _read_gmr(self):
        import pathlib
        return pathlib.Path(
            "/export/home2/broc/vllm-thought-eviction/vllm/v1/worker/gpu_model_runner.py"
        ).read_text()

    def test_gpu_model_runner_has_enable_l2_norms_at_three_sites(self):
        """enable_l2_norms appears at attn_metadata build, forward-pass call site, and
        _compute_l2_norms inner loop — at least 3 occurrences."""
        src = self._read_gmr()
        count = src.count("enable_l2_norms")
        self.assertGreaterEqual(
            count, 3,
            f"Expected >=3 occurrences of enable_l2_norms in gpu_model_runner, got {count}",
        )

    def test_no_cross_process_singleton_guards(self):
        """has_any_fetch_pending() must not appear as a guard in gpu_model_runner."""
        src = self._read_gmr()
        self.assertNotIn(
            "has_any_fetch_pending()",
            src,
            "gpu_model_runner must not use has_any_fetch_pending() (cross-process anti-pattern)",
        )

    def test_no_has_any_active_guard(self):
        """has_any_active() must not appear as a guard in gpu_model_runner."""
        src = self._read_gmr()
        self.assertNotIn(
            "has_any_active()",
            src,
            "gpu_model_runner must not use has_any_active() (cross-process anti-pattern)",
        )


class TestIpcOverhead(unittest.TestCase):
    """GUARD-02: IPC serialization size is unchanged for non-eviction requests."""

    def test_engine_core_output_norms_omitted_when_none(self):
        """EngineCoreOutput with new_l2_norms=None serializes smaller than one with norms."""
        from vllm.v1.engine import EngineCoreOutput
        without_norms = EngineCoreOutput(request_id="req-std", new_token_ids=[42])
        with_norms = EngineCoreOutput(
            request_id="req-evict", new_token_ids=[42], new_l2_norms=[0.5, 0.6, 0.7]
        )
        encoded_without = msgspec.msgpack.encode(without_norms)
        encoded_with = msgspec.msgpack.encode(with_norms)
        self.assertGreater(
            len(encoded_with), len(encoded_without),
            "Output with norms must serialize larger than output without (None is omitted)",
        )


if __name__ == "__main__":
    unittest.main()
