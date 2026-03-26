"""
Integration tests for EvictionOrchestrator activation in serving.py.

Coverage:
- Test 1: EvictionOrchestrator is imported in serving module
- Test 2: Conditional activation pattern exists in create_chat_completion
- Test 3: block_size is not hardcoded — sourced from cache_config
- Test 4: request_id passed without double-prefix (uses existing chatcmpl- prefixed id)
- Test 5: wrap_stream call is present in the activation code path
"""

import inspect
import re


# ---------------------------------------------------------------------------
# Test 1: Import is present in serving module
# ---------------------------------------------------------------------------


def test_orchestrator_import_in_serving():
    """Verify serving.py imports EvictionOrchestrator at module level."""
    import importlib
    mod = importlib.import_module(
        'vllm.entrypoints.openai.chat_completion.serving'
    )
    assert hasattr(mod, 'EvictionOrchestrator'), (
        "EvictionOrchestrator not found in serving module — import missing"
    )


# ---------------------------------------------------------------------------
# Test 2: Conditional activation pattern in create_chat_completion
# ---------------------------------------------------------------------------


def test_orchestrator_activation_code_present():
    """Verify the conditional activation pattern exists in serving.py source."""
    from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
    source = inspect.getsource(OpenAIServingChat.create_chat_completion)
    assert 'request.eviction_params is not None' in source, (
        "Missing eviction_params guard in create_chat_completion"
    )
    assert 'EvictionOrchestrator(' in source, (
        "Missing EvictionOrchestrator constructor call"
    )
    assert 'wrap_stream' in source, (
        "Missing wrap_stream call in create_chat_completion"
    )
    assert 'cache_config.block_size' in source, (
        "Missing cache_config.block_size reference — block_size may be hardcoded"
    )


# ---------------------------------------------------------------------------
# Test 3: block_size is NOT hardcoded
# ---------------------------------------------------------------------------


def test_block_size_not_hardcoded():
    """Verify block_size comes from config, not a numeric literal."""
    from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
    source = inspect.getsource(OpenAIServingChat.create_chat_completion)
    # Should NOT have block_size=16 or block_size=32 etc. as a literal assignment
    hardcoded = re.findall(r'block_size\s*=\s*\d+', source)
    assert len(hardcoded) == 0, (
        f"Found hardcoded block_size literal(s): {hardcoded}"
    )


# ---------------------------------------------------------------------------
# Test 4: request_id is not double-prefixed
# ---------------------------------------------------------------------------


def test_request_id_not_double_prefixed():
    """Verify orchestrator receives request_id directly (not re-prefixed).

    The request_id is already f"chatcmpl-{uuid}" at construction time (line
    341-343 of serving.py). The orchestrator should receive this value as-is,
    not wrapped in another f"chatcmpl-{request_id}" string.
    """
    from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
    source = inspect.getsource(OpenAIServingChat.create_chat_completion)
    # The pattern request_id=request_id (with optional spaces) must be present
    normalized = source.replace(' ', '')
    assert 'request_id=request_id' in normalized, (
        "request_id not passed directly to EvictionOrchestrator — "
        "check for re-prefixing or renaming"
    )
    # Must NOT re-prefix request_id with "chatcmpl-" when passing to orchestrator.
    # Pattern: f"chatcmpl-{request_id}" or "chatcmpl-" + request_id as a kwarg.
    double_prefix_in_call = re.findall(
        r'request_id\s*=\s*f["\']chatcmpl-\{request_id\}',
        source
    )
    assert len(double_prefix_in_call) == 0, (
        f"request_id is being double-prefixed near orchestrator: {double_prefix_in_call}"
    )


# ---------------------------------------------------------------------------
# Test 5: wrap_stream wires the generator into the stream path
# ---------------------------------------------------------------------------


def test_wrap_stream_wires_result_generator():
    """Verify result_generator is reassigned from orchestrator.wrap_stream."""
    from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
    source = inspect.getsource(OpenAIServingChat.create_chat_completion)
    # result_generator should be reassigned to orchestrator.wrap_stream(result_generator)
    assert 'result_generator = orchestrator.wrap_stream(result_generator)' in source, (
        "result_generator not reassigned from orchestrator.wrap_stream() — "
        "the wrapped generator won't reach chat_completion_stream_generator"
    )
