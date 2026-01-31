"""PagedEviction module for vLLM v1 engine.

This module provides L2-norm based KV cache eviction functionality,
ported from vLLM-PagedEviction for the v1 engine with FlashAttention.

Main components:
- PagedEvictConfig: Configuration for eviction parameters
- PagedEvictionManager: High-level manager coordinating eviction
- PagedEvictionIntegration: Integration layer for GPUModelRunner
- KVCachePruner: Core scoring and pruning logic
- BlockL2NormManager: Per-request block score tracking
"""

from vllm.v1.core.paged_evict_config import PagedEvictConfig
from vllm.v1.core.paged_eviction_manager import PagedEvictionManager
from vllm.v1.core.page_evict_utils import (
    get_num_required_blocks_after_prune_prompt,
    calculate_tokens_to_prune,
    calculate_blocks_to_free,
    get_evictable_block_range,
    should_trigger_eviction,
)
from vllm.v1.attention.kvcache_pruner import (
    KVCachePruner,
    compute_block_l2norms_from_cache,
)
from vllm.v1.attention.block_l2norm_manager import BlockL2NormManager
from vllm.v1.paged_eviction.integration import (
    PagedEvictionIntegration,
    EvictionResult,
)

__all__ = [
    # Config
    "PagedEvictConfig",
    # Managers
    "PagedEvictionManager",
    "PagedEvictionIntegration",
    # Pruner
    "KVCachePruner",
    "compute_block_l2norms_from_cache",
    # L2 Norm Manager
    "BlockL2NormManager",
    # Utilities
    "get_num_required_blocks_after_prune_prompt",
    "calculate_tokens_to_prune",
    "calculate_blocks_to_free",
    "get_evictable_block_range",
    "should_trigger_eviction",
    # Result types
    "EvictionResult",
]
