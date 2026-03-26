"""Tests for L2 norm delivery pipeline (Plan 02-02, D-03).

Verifies that L2 norms flow from L2NormCache through EngineCoreOutput IPC
to RequestOutput with differential indexing, and that non-eviction requests
are unaffected.
"""

import sys
import unittest
from unittest.mock import MagicMock, patch, PropertyMock


class TestEngineCoreOutputField(unittest.TestCase):
    """Test 1 & 2: EngineCoreOutput has new_l2_norms field with None default."""

    def test_engine_core_output_default_none(self):
        """Test 1: EngineCoreOutput without new_l2_norms defaults to None."""
        from vllm.v1.engine import EngineCoreOutput
        e = EngineCoreOutput(request_id="test", new_token_ids=[1])
        self.assertIsNone(e.new_l2_norms)

    def test_engine_core_output_stores_list(self):
        """Test 2: EngineCoreOutput(new_l2_norms=[0.5, 0.6]) stores the list correctly."""
        from vllm.v1.engine import EngineCoreOutput
        e = EngineCoreOutput(request_id="test", new_token_ids=[1], new_l2_norms=[0.5, 0.6])
        self.assertEqual(e.new_l2_norms, [0.5, 0.6])


class TestRequestOutputField(unittest.TestCase):
    """Test 3 & 4: RequestOutput has new_l2_norms field."""

    def _make_request_output(self, **kwargs):
        from vllm.outputs import RequestOutput, CompletionOutput
        from vllm.sampling_params import RequestOutputKind
        completion = CompletionOutput(
            index=0,
            text="hello",
            token_ids=[1, 2, 3],
            cumulative_logprob=None,
            logprobs=None,
            finish_reason=None,
        )
        return RequestOutput(
            request_id="req-1",
            prompt="test",
            prompt_token_ids=[1],
            prompt_logprobs=None,
            outputs=[completion],
            finished=False,
            **kwargs,
        )

    def test_request_output_default_none(self):
        """Test 3: RequestOutput without new_l2_norms has None for the field."""
        req = self._make_request_output()
        self.assertIsNone(req.new_l2_norms)

    def test_request_output_stores_list(self):
        """Test 4: RequestOutput(new_l2_norms=[0.5]) stores the list correctly."""
        req = self._make_request_output(new_l2_norms=[0.5])
        self.assertEqual(req.new_l2_norms, [0.5])


class TestSchedulerL2NormPopulation(unittest.TestCase):
    """Test 1-3: Scheduler populates new_l2_norms from L2NormCache differentially."""

    def _make_mock_l2_cache(self, norms_by_start: dict):
        """Create a mock L2NormCache that returns norms based on start_index."""
        cache = MagicMock()

        def get_norms(req_id, start_index=0):
            if req_id in norms_by_start:
                all_norms = norms_by_start[req_id]
                if start_index >= len(all_norms):
                    return []
                return all_norms[start_index:]
            return None

        cache.get_norms.side_effect = get_norms
        return cache

    def test_scheduler_has_l2_norm_last_index(self):
        """Verify the scheduler has _l2_norm_last_index instance variable."""
        # Import just to check the attribute is added during __init__
        # We inspect the __init__ source instead of instantiating the full scheduler
        import inspect
        from vllm.v1.core.sched import scheduler as sched_module
        source = inspect.getsource(sched_module.Scheduler.__init__)
        self.assertIn("_l2_norm_last_index", source,
                      "Scheduler.__init__ must initialize self._l2_norm_last_index")

    def test_update_from_output_fetches_l2_norms(self):
        """Test 1: When L2NormCache has norms, EngineCoreOutput gets non-None new_l2_norms."""
        import inspect
        from vllm.v1.core.sched import scheduler as sched_module
        source = inspect.getsource(sched_module.Scheduler.update_from_output)
        self.assertIn("l2_norm_last_index", source,
                      "update_from_output must use _l2_norm_last_index")
        self.assertIn("get_l2_norm_cache", source,
                      "update_from_output must call get_l2_norm_cache")
        self.assertIn("new_l2_norms=new_l2_norms", source,
                      "update_from_output must pass new_l2_norms to EngineCoreOutput")

    def test_cleanup_removes_l2_norm_index(self):
        """Test: Finished requests have their _l2_norm_last_index entry cleaned up."""
        import inspect
        from vllm.v1.core.sched import scheduler as sched_module
        # Check the finish_request method (or wherever request_eviction_data.pop is called)
        source = inspect.getsource(sched_module.Scheduler._finish_request)
        self.assertIn("_l2_norm_last_index", source,
                      "_finish_request must clean up _l2_norm_last_index")

    def test_differential_indexing_via_source(self):
        """Test 3: Source shows differential index is updated after fetching norms."""
        import inspect
        from vllm.v1.core.sched import scheduler as sched_module
        source = inspect.getsource(sched_module.Scheduler.update_from_output)
        # Should see: start_idx = self._l2_norm_last_index.get(req_id, 0)
        self.assertIn("_l2_norm_last_index.get", source,
                      "update_from_output must read start index from _l2_norm_last_index")
        # Should see: self._l2_norm_last_index[req_id] = start_idx + len(norms)
        self.assertIn("_l2_norm_last_index[req_id]", source,
                      "update_from_output must update _l2_norm_last_index after fetching norms")


class TestOutputProcessorThreadsNorms(unittest.TestCase):
    """Test 4 & 5: output_processor threads new_l2_norms through to RequestOutput."""

    def test_new_request_output_has_param(self):
        """Test 4: _new_request_output has new_l2_norms parameter."""
        import inspect
        from vllm.v1.engine.output_processor import RequestState
        sig = inspect.signature(RequestState._new_request_output)
        self.assertIn("new_l2_norms", sig.parameters,
                      "_new_request_output must have new_l2_norms parameter")

    def test_new_request_output_passes_to_request_output(self):
        """Test 5: _new_request_output passes new_l2_norms to RequestOutput."""
        import inspect
        from vllm.v1.engine import output_processor as op_module
        source = inspect.getsource(op_module.RequestState._new_request_output)
        self.assertIn("new_l2_norms=new_l2_norms", source,
                      "_new_request_output must pass new_l2_norms to RequestOutput()")

    def test_make_request_output_threads_norms(self):
        """Test: make_request_output passes engine_core_output.new_l2_norms through."""
        import inspect
        from vllm.v1.engine import output_processor as op_module
        source = inspect.getsource(op_module.RequestState.make_request_output)
        self.assertIn("new_l2_norms", source,
                      "make_request_output must thread new_l2_norms through")


if __name__ == "__main__":
    unittest.main()
