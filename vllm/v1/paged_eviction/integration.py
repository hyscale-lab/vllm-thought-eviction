# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
PagedEviction Integration Layer for vLLM v1 Engine.

This module provides the integration hooks between PagedEviction logic and
the v1 engine core components (GPUModelRunner, Scheduler).

Integration Points:
1. After prefill: Prune KV cache tokens before storing (optional aggressive pruning)
2. After decode steps: Check if budget exceeded and evict blocks
3. L2-norm computation: Compute norms from KV cache after attention

Usage in GPUModelRunner:
    from vllm.v1.paged_eviction.integration import PagedEvictionIntegration

    # In __init__:
    self.eviction_integration = PagedEvictionIntegration(
        paged_eviction_manager=scheduler.paged_eviction_manager,
        kv_cache_manager=kv_cache_manager,
        block_size=block_size,
    )

    # After model forward in execute_model:
    if self.eviction_integration.enabled:
        evicted_blocks = self.eviction_integration.maybe_evict(
            request_id=req_id,
            num_blocks_used=num_blocks,
            kv_cache=kv_cache,
            layer_idx=0,  # Typically use first layer's norms
        )
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.paged_eviction_manager import PagedEvictionManager

logger = logging.getLogger(__name__)


@dataclass
class EvictionResult:
    """Result of an eviction operation."""
    request_id: str
    evicted_block_ids: list[int]
    num_evicted_tokens: int
    remaining_budget: int


class PagedEvictionIntegration:
    """
    Integration layer for PagedEviction with v1 engine components.
    
    This class provides high-level methods for:
    - Checking if eviction is needed based on cache usage
    - Triggering eviction and returning block IDs to free
    - Computing L2-norms from KV cache tensors
    - Prefill-time token pruning
    """
    
    def __init__(
        self,
        paged_eviction_manager: "PagedEvictionManager | None",
        kv_cache_manager: "KVCacheManager | None" = None,
        block_size: int = 16,
    ):
        """
        Initialize the integration layer.
        
        Args:
            paged_eviction_manager: The PagedEvictionManager from scheduler.
                If None, eviction is disabled.
            kv_cache_manager: The KVCacheManager for block operations.
            block_size: Number of tokens per block.
        """
        self.paged_eviction_manager = paged_eviction_manager
        self.kv_cache_manager = kv_cache_manager
        self.block_size = block_size
        self.enabled = paged_eviction_manager is not None
        
        if self.enabled:
            logger.info(
                "PagedEvictionIntegration enabled with manager: %s",
                type(paged_eviction_manager).__name__
            )
    
    def should_evict(
        self,
        request_id: str,
        num_blocks_used: int,
    ) -> bool:
        """
        Check if eviction should be triggered for a request.
        
        Args:
            request_id: The request ID to check.
            num_blocks_used: Current number of blocks used by the request.
            
        Returns:
            True if eviction is needed, False otherwise.
        """
        if not self.enabled:
            return False
        
        assert self.paged_eviction_manager is not None
        # Convert blocks to tokens for the manager's interface
        current_tokens = num_blocks_used * self.block_size
        return self.paged_eviction_manager.should_evict(
            request_id=request_id,
            current_tokens=current_tokens,
        )
    
    def maybe_evict(
        self,
        request_id: str,
        num_blocks_used: int,
        kv_cache: torch.Tensor | None = None,
        block_ids: list[int] | None = None,
        layer_idx: int = 0,
    ) -> EvictionResult | None:
        """
        Check if eviction needed and perform it if so.
        
        This method:
        1. Checks if the request exceeds its cache budget
        2. If so, computes L2-norms from KV cache (if provided)
        3. Selects blocks to evict based on lowest scores
        4. Returns the evicted block information
        
        Args:
            request_id: The request ID to potentially evict from.
            num_blocks_used: Current number of blocks used.
            kv_cache: Optional KV cache tensor for L2-norm computation.
                Shape: [2, num_blocks, block_size, num_kv_heads, head_size]
            block_ids: Optional list of block IDs currently allocated to request.
            layer_idx: Layer index for L2-norm computation (default: 0).
            
        Returns:
            EvictionResult if eviction occurred, None otherwise.
        """
        if not self.enabled:
            return None
        
        if not self.should_evict(request_id, num_blocks_used):
            return None
        
        assert self.paged_eviction_manager is not None
        
        # Compute L2-norms from KV cache if provided
        if kv_cache is not None and block_ids is not None:
            self._update_block_l2norms(
                request_id=request_id,
                kv_cache=kv_cache,
                block_ids=block_ids,
                layer_idx=layer_idx,
            )
        
        # Get blocks to evict
        blocks_to_evict = self.paged_eviction_manager.get_blocks_to_evict(
            request_id=request_id,
            num_blocks=num_blocks_used,
            block_ids=block_ids or [],
        )
        
        if not blocks_to_evict:
            return None
        
        # Free the blocks via KVCacheManager if available
        if self.kv_cache_manager is not None:
            self._free_evicted_blocks(request_id, blocks_to_evict)
        
        num_evicted_tokens = len(blocks_to_evict) * self.block_size
        config = self.paged_eviction_manager.paged_evict_config
        remaining_budget = config.cache_budget - (
            (num_blocks_used - len(blocks_to_evict)) * self.block_size
        )
        
        logger.debug(
            "Evicted %d blocks from request %s: %s",
            len(blocks_to_evict),
            request_id,
            blocks_to_evict,
        )
        
        return EvictionResult(
            request_id=request_id,
            evicted_block_ids=blocks_to_evict,
            num_evicted_tokens=num_evicted_tokens,
            remaining_budget=remaining_budget,
        )
    
    def prune_prefill_kv(
        self,
        request_id: str,
        key: torch.Tensor,
        value: torch.Tensor,
        num_prompt_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        """
        Prune prefill K,V tensors before caching.
        
        This method applies prefill-time pruning to reduce the number of
        tokens stored in the KV cache. It computes L2-norms and selects
        tokens to keep based on the eviction method's scoring function.
        
        Args:
            request_id: The request ID for tracking.
            key: Key tensor of shape [num_prompt_tokens, num_kv_heads, head_size]
            value: Value tensor of shape [num_prompt_tokens, num_kv_heads, head_size]
            num_prompt_tokens: Number of tokens in the prompt.
            
        Returns:
            Tuple of (pruned_key, pruned_value, kept_token_indices)
        """
        if not self.enabled:
            return key, value, list(range(num_prompt_tokens))
        
        assert self.paged_eviction_manager is not None
        
        pruned_key, pruned_value, kept_indices = (
            self.paged_eviction_manager.prune_prefill_kv(
                request_id=request_id,
                key=key,
                value=value,
                num_tokens=num_prompt_tokens,
            )
        )
        
        if len(kept_indices) < num_prompt_tokens:
            logger.debug(
                "Pruned prefill for request %s: %d -> %d tokens",
                request_id,
                num_prompt_tokens,
                len(kept_indices),
            )
        
        return pruned_key, pruned_value, kept_indices
    
    def _update_block_l2norms(
        self,
        request_id: str,
        kv_cache: torch.Tensor,
        block_ids: list[int],
        layer_idx: int,
    ) -> None:
        """
        Compute and update L2-norms for blocks from KV cache.
        
        Args:
            request_id: The request ID.
            kv_cache: KV cache tensor.
            block_ids: List of block IDs to compute norms for.
            layer_idx: Layer index.
        """
        assert self.paged_eviction_manager is not None
        
        # Skip if no block IDs
        if not block_ids:
            return
        
        # Compute L2 norms directly from KV cache
        # kv_cache shape: [2, num_blocks, block_size, num_kv_heads, head_size]
        key_cache = kv_cache[0]  # [num_blocks, block_size, num_kv_heads, head_size]
        value_cache = kv_cache[1]
        
        for block_id in block_ids:
            if block_id < 0 or block_id >= key_cache.shape[0]:
                continue
            
            # Get key and value for this block
            key_block = key_cache[block_id]  # [block_size, num_kv_heads, head_size]
            value_block = value_cache[block_id]
            
            # Compute L2 norms
            key_l2 = torch.norm(key_block.float(), dim=-1).mean().item()
            value_l2 = torch.norm(value_block.float(), dim=-1).mean().item()
            
            # Update in manager
            if key_l2 > 0:
                score = value_l2 / key_l2
            else:
                score = float('inf')  # Avoid division by zero
            
            self.paged_eviction_manager.l2norm_manager.update_block_score(
                request_id=request_id,
                layer_idx=layer_idx,
                block_idx=block_id,
                score=score,
            )
    
    def _free_evicted_blocks(
        self,
        request_id: str,
        block_ids: list[int],
    ) -> None:
        """
        Free evicted blocks via the KVCacheManager.
        
        Args:
            request_id: The request ID.
            block_ids: List of block IDs to free.
        """
        # This would call the actual block freeing logic
        # For now, just log the action since direct block freeing
        # requires more integration with KVCacheManager internals
        logger.debug(
            "Would free blocks for request %s: %s",
            request_id,
            block_ids,
        )
    
    def get_stats(self) -> dict[str, Any]:
        """Get eviction statistics."""
        if not self.enabled:
            return {"enabled": False}
        
        assert self.paged_eviction_manager is not None
        stats = self.paged_eviction_manager.get_stats()
        stats["enabled"] = True
        return stats
