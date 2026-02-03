# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
PagedEviction Integration Layer for vLLM v1 Engine.

This module provides the integration hooks between PagedEviction logic and
the v1 engine core components (GPUModelRunner, Scheduler).

Integration Points:
1. After prefill: Prune KV cache tokens before storing (optional aggressive pruning)
2. After decode steps: Check if budget exceeded and compute metrics for eviction
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
        eviction_stats = self.eviction_integration.compute_eviction_metrics(
            request_id=req_id,
            num_blocks_used=num_blocks,
            kv_cache=kv_cache,
            layer_idx=0,  # Typically use first layer's norms
        )
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheManager
    from vllm.v1.core.paged_evict_config import PagedEvictConfig
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
        paged_eviction_manager: "PagedEvictionManager | None" = None,
        kv_cache_manager: "KVCacheManager | None" = None,
        block_size: int = 16,
        paged_evict_config: "PagedEvictConfig | None" = None,
    ):
        """
        Initialize the integration layer.
        
        Args:
            paged_eviction_manager: The PagedEvictionManager from scheduler.
            kv_cache_manager: The KVCacheManager for block operations.
            block_size: Number of tokens per block.
            paged_evict_config: Configuration for eviction (used if manager is None).
        """
        self.paged_eviction_manager = paged_eviction_manager
        self.kv_cache_manager = kv_cache_manager
        self.block_size = block_size
        self.paged_evict_config = paged_evict_config
        
        # Enabled if either a manager is provided OR a config is provided
        self.enabled = (paged_eviction_manager is not None) or (paged_evict_config is not None)
        
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
            logger.debug(
                "[EVICTION DEBUG] integration.should_evict: eviction not enabled"
            )
            return False
        
        # Use manager if available
        if self.paged_eviction_manager is not None:
            # Convert blocks to tokens for the manager's interface
            current_tokens = num_blocks_used * self.block_size
            return self.paged_eviction_manager.should_evict(
                request_id=request_id,
                current_tokens=current_tokens,
            )
            
        # Fallback to config if available (Worker-side check)
        if self.paged_evict_config is not None:
             # Simple budget check based on config
             cache_budget = self.paged_evict_config.cache_budget
             if cache_budget is not None:
                 if num_blocks_used * self.block_size > cache_budget:
                     return True
             
             # Ratio based check is harder without global knowledge, 
             # but we can assume we should compute stats if ratio is enabled
             # so that Scheduler can make decisions.
             if self.paged_evict_config.cache_budget_ratio is not None:
                 return True # Always compute stats for ratio mode
                 
        return False
    
    def compute_eviction_metrics(
        self,
        request_id: str,
        num_blocks_used: int,
        kv_cache: torch.Tensor | None = None,
        block_ids: list[int] | None = None,
        layer_idx: int = 0,
    ) -> Dict[str, Dict[int, List[float]]] | None:
        """
        Check if eviction needed and compute metrics if so.
        
        Args:
            request_id: The request ID to potentially evict from.
            num_blocks_used: Current number of blocks used.
            kv_cache: Optional KV cache tensor for L2-norm computation.
                Shape: [2, num_blocks, block_size, num_kv_heads, head_size]
            block_ids: Optional list of block IDs currently allocated to request.
            layer_idx: Layer index for L2-norm computation (default: 0).
            
        Returns:
            Dictionary mapping request_id -> layer_idx -> list of scores, or None if no metrics.
        """
        if not self.enabled:
            return None
        
        if not self.should_evict(request_id, num_blocks_used):
            return None
        
        # Compute L2-norms from KV cache if provided
        if kv_cache is not None and block_ids is not None:
            scores = self._compute_block_l2norms(
                kv_cache=kv_cache,
                block_ids=block_ids,
            )
            # Map block_ids to scores? No, we need scores for all blocks in block_ids order
            # The manager expects a list corresponding to blocks?
            # Actually, `BlockL2NormManager.update_block_score` takes `block_idx`.
            # But here we are returning scores map.
            # To be efficient, we return {request_id: {layer_idx: [scores_ordered_by_block_ID]}}?
            # Or {request_id: {layer_idx: [(block_id, score), ...]}}?
            # Let's align with what BlockL2NormManager expects.
            # It expects `update_block_score(req, layer, block_idx, score)`.
            # So let's return a sparse map: {block_id: score} for simplicity?
            # But the proposed signature in output is `list[float]`.
            # Ah, `BlockView` in manager usually tracks logical blocks.
            # But here we have physical block IDs.
            # The Scheduler knows mapping Logical -> Physical.
            
            # Wait, `BlockL2NormManager` stores norms per `block_idx`.
            # If `block_idx` corresponds to PHYSICAL block ID, then we need to send that.
            # If it's LOGICAL, we need to map.
            # The `block_ids` passed here are PHYSICAL.
            # So let's verify if `BlockL2NormManager` works with Physical or Logical.
            # `BlockL2NormManager` doesn't seem to distinquish, just int keys.
            # In `paged_eviction_manager.py`, `get_blocks_to_evict` uses `block_ids`.
            # `block_ids` are passed from Scheduler.
            # Scheduler passes physical ones?
            # Let's assume Physical Block IDs are the keys.
            
            # Since `outputs.py` defined `list[float]`, let's check definition:
            # `paged_eviction_stats: dict[str, dict[int, list[float]]]`
            # A list implies indexed by... something.
            # If it's sparse updates, Dict[int, float] would be better.
            # But `outputs` says `list[float]`.
            # Maybe I should change `outputs` to `Dict[int, float]` to be safe?
            # But `BlockL2NormManager` seems to use `List[float]` in `reqid_mapping_block_l2norms`.
            # Re-checking `BlockL2NormManager`:
            # `self.reqid_mapping_block_l2norms: Optional[Dict[str, Dict[int, List[float]]]]`
            # It uses List! This implies it expects dense arrays indexed by Logical Block Index?
            # Or is it a list where index matches `reqid_mapping_blocktables`?
            # If it maps `req_id -> layer -> list`, then index i corresponds to block i in logical sequence?
            
            # Re-read `BlockL2NormManager`... 
            # `get_lowest_scoring_block` iterates `range(initial_blocks, num_blocks - 1)`.
            # This strongly implies LOGICAL block indexing.
            
            # Issue: Worker operates on PHYSICAL blocks (`block_table`).
            # Worker doesn't know Logical mappings easily (unless passed).
            # `input_batch` has `block_table`.
            # `gpu_model_runner` extracts `current_block_ids` from `block_table`.
            # These are physical.
            # The order in `block_table` row IS the logical order!
            # So `block_ids[i]` is the physical ID of logical block `i`.
            # Perfect.
            
            # So we just compute norms for `block_ids` in order, and return that list.
            # The Scheduler will receive `List[float]`, which corresponds to logical blocks 0..N.
            # Then it updates `reqid_mapping_block_l2norms` with this list.
            
            return {request_id: {layer_idx: scores}}

        return None

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
        
        return pruned_key, pruned_value, kept_indices
    
    def _compute_block_l2norms(
        self,
        kv_cache: torch.Tensor,
        block_ids: list[int],
    ) -> List[float]:
        """
        Compute L2-norms for blocks from KV cache.
        
        Args:
            kv_cache: KV cache tensor.
            block_ids: List of block IDs to compute norms for.
            
        Returns:
             List of scores corresponding to block_ids order.
        """
        # Compute L2 norms directly from KV cache
        # kv_cache shape: [2, num_blocks, block_size, num_kv_heads, head_size]
        key_cache = kv_cache[0]  # [num_blocks, block_size, num_kv_heads, head_size]
        value_cache = kv_cache[1]
        
        scores = []
        for block_id in block_ids:
            if block_id < 0 or block_id >= key_cache.shape[0]:
                scores.append(0.0) # Should not happen?
                continue
            
            # Get key and value for this block
            key_block = key_cache[block_id]  # [block_size, num_kv_heads, head_size]
            value_block = value_cache[block_id]
            
            # Compute L2 norms
            key_l2 = torch.norm(key_block.float(), dim=-1).mean().item()
            value_l2 = torch.norm(value_block.float(), dim=-1).mean().item()
            
            # Score
            if key_l2 > 0:
                score = value_l2 / key_l2
            else:
                score = float('inf')  # Avoid division by zero
            
            scores.append(score)
            
        return scores
    
    def get_stats(self) -> dict[str, Any]:
        """Get eviction statistics."""
        if not self.enabled:
            return {"enabled": False}
        
        assert self.paged_eviction_manager is not None
        stats = self.paged_eviction_manager.get_stats()
        stats["enabled"] = True
        return stats
