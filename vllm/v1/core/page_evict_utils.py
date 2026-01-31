"""Utility functions for PagedEviction in vLLM v1 engine.

Provides helper functions for calculating block requirements after pruning.
Ported from vLLM-PagedEviction.
"""

import math
from typing import Optional

from vllm.v1.core.paged_evict_config import PagedEvictConfig


def get_num_required_blocks_after_prune_prompt(
    q_len: int, 
    paged_evict_config: PagedEvictConfig, 
    block_size: int
) -> tuple[int, int]:
    """Calculate the number of blocks required after prompt pruning.
    
    Determines how many blocks will be needed after applying the cache budget
    constraint during prefill.
    
    Args:
        q_len: Query/prompt length in tokens
        paged_evict_config: PagedEviction configuration
        block_size: Number of tokens per block
        
    Returns:
        Tuple of (num_blocks, total_unpruned_tokens)
    """
    # Helper for ceiling division
    def cdiv(a: int, b: int) -> int:
        return (a + b - 1) // b
    
    # If under budget, no pruning needed
    if q_len < paged_evict_config.cache_budget:
        return cdiv(q_len, block_size), q_len
    
    # Calculate slices
    end_first_slice_idx = paged_evict_config.initial_blocks * block_size
    remainder_size = q_len % block_size
    
    # End of middle slice (before last block and remainder)
    end_middle_slice_idx = q_len - block_size - remainder_size
    
    # Number of middle tokens available
    num_middle_slice_tokens = end_middle_slice_idx - end_first_slice_idx
    
    # Handle different eviction methods
    if paged_evict_config.evict_method in [
        "streamingLLM", "streamingLLM-1", 
        "inverse_key_l2", "global", "local",
        "value_l2_div_key_l2", "value_l2", "key_l2"
    ]:
        if paged_evict_config.cache_prune_type == "budget":
            # Calculate how many middle tokens to keep
            middle_unpruned_tokens = paged_evict_config.cache_budget - \
                (paged_evict_config.initial_blocks * block_size) - block_size
            
            # Ensure non-negative
            middle_unpruned_tokens = max(0, middle_unpruned_tokens)
            
            # Total unpruned = first slice + middle unpruned + last slice
            total_unpruned_tokens = end_first_slice_idx + middle_unpruned_tokens + \
                (block_size + remainder_size)
            
            # Clamp to original length
            total_unpruned_tokens = min(total_unpruned_tokens, q_len)
            
            return cdiv(total_unpruned_tokens, block_size), total_unpruned_tokens
        else:
            raise ValueError(
                f"Unsupported cache_prune_type: {paged_evict_config.cache_prune_type} "
                f"for evict_method: {paged_evict_config.evict_method}"
            )
    else:
        raise ValueError(
            f"Unsupported eviction_method: {paged_evict_config.evict_method} "
            f"for cache_prune_type: {paged_evict_config.cache_prune_type}"
        )


def calculate_tokens_to_prune(
    seq_len: int,
    cache_budget: int,
    block_size: int,
    initial_blocks: int = 1
) -> int:
    """Calculate number of tokens to prune to meet cache budget.
    
    Args:
        seq_len: Current sequence length
        cache_budget: Target cache budget in tokens
        block_size: Number of tokens per block
        initial_blocks: Number of initial blocks to preserve
        
    Returns:
        Number of tokens to prune (0 if under budget)
    """
    if seq_len <= cache_budget:
        return 0
    
    return seq_len - cache_budget


def calculate_blocks_to_free(
    current_blocks: int,
    target_tokens: int,
    block_size: int
) -> int:
    """Calculate number of blocks to free to reach target token count.
    
    Args:
        current_blocks: Current number of allocated blocks
        target_tokens: Target number of tokens
        block_size: Number of tokens per block
        
    Returns:
        Number of blocks to free
    """
    target_blocks = (target_tokens + block_size - 1) // block_size
    return max(0, current_blocks - target_blocks)


def get_evictable_block_range(
    total_blocks: int,
    initial_blocks: int = 1,
    final_blocks: int = 1
) -> tuple[int, int]:
    """Get the range of block indices that can be evicted.
    
    Excludes initial sink blocks and final context blocks.
    
    Args:
        total_blocks: Total number of blocks
        initial_blocks: Number of initial blocks to preserve
        final_blocks: Number of final blocks to preserve
        
    Returns:
        Tuple of (start_block_idx, end_block_idx) - exclusive end
    """
    start_idx = initial_blocks
    end_idx = max(start_idx, total_blocks - final_blocks)
    
    return start_idx, end_idx


def should_trigger_eviction(
    current_tokens: int,
    cache_budget: int,
    block_size: int
) -> bool:
    """Check if eviction should be triggered based on current usage.
    
    Args:
        current_tokens: Current number of tokens in cache
        cache_budget: Maximum allowed tokens
        block_size: Block size for alignment check
        
    Returns:
        True if eviction should be triggered
    """
    return current_tokens > cache_budget
