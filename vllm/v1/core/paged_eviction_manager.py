"""Paged Eviction Manager for vLLM v1 engine.

Coordinates L2-norm based KV cache eviction using PagedEviction methodology.
Integrates with existing v1 KVCacheManager and BlockPool infrastructure.

This manager:
1. Tracks L2-norm scores for blocks across layers
2. Determines which blocks to evict when over budget
3. Triggers eviction through KVCacheManager

Ported from vLLM-PagedEviction for v1 engine.
"""

from typing import Optional, Dict, List, Set
import torch

from vllm.logger import init_logger
from vllm.v1.attention.kvcache_pruner import KVCachePruner, compute_block_l2norms_from_cache
from vllm.v1.attention.block_l2norm_manager import BlockL2NormManager
from vllm.v1.core.paged_evict_config import PagedEvictConfig
from vllm.v1.core.page_evict_utils import (
    get_num_required_blocks_after_prune_prompt,
    should_trigger_eviction,
    get_evictable_block_range,
)

logger = init_logger(__name__)


class PagedEvictionManager:
    """Manages score-based KV cache eviction for v1 engine.
    
    Coordinates between L2-norm scoring, block management, and the eviction
    decision process. Designed to work with FlashAttention backend where
    L2 norms are computed from cached KV values.
    
    Args:
        paged_evict_config: Configuration for paged eviction
        block_size: Number of tokens per KV cache block
        num_layers: Number of transformer layers
    """
    
    def __init__(
        self,
        paged_evict_config: PagedEvictConfig,
        block_size: int,
        num_layers: int,
    ):
        self.config = paged_evict_config
        self.block_size = block_size
        self.num_layers = num_layers
        
        # Initialize pruner for scoring
        # For prefill pruning, use fixed budget or a reasonable default if ratio mode
        pruner_budget = paged_evict_config.cache_budget or 4096  # Default for ratio mode
        self.pruner = KVCachePruner(
            cache_prune_type=paged_evict_config.cache_prune_type,
            evict_method=paged_evict_config.evict_method,
            block_size=block_size,
            cache_budget=pruner_budget,
            initial_blocks=paged_evict_config.initial_blocks,
        )
        
        # Block L2-norm tracking
        self.l2norm_manager = BlockL2NormManager()
        
        # Per-request block tracking
        # request_id -> list of block_ids per layer
        self.request_blocks: Dict[str, Dict[int, List[int]]] = {}
        
        # Evicted block IDs per request (to avoid re-evicting)
        self.evicted_blocks: Dict[str, Set[int]] = {}
        
        # Cumulative evicted blocks counter (persists across requests)
        self.cumulative_evicted_blocks: int = 0
        
        logger.info(
            f"PagedEvictionManager initialized: "
            f"{'ratio=' + str(paged_evict_config.cache_budget_ratio) if paged_evict_config.cache_budget_ratio else 'budget=' + str(paged_evict_config.cache_budget)}, "
            f"method={paged_evict_config.evict_method}, block_size={block_size}"
        )
    
    def register_request(self, request_id: str) -> None:
        """Register a new request for eviction tracking.
        
        Args:
            request_id: ID of the new request
        """
        if request_id not in self.request_blocks:
            self.request_blocks[request_id] = {}
            self.evicted_blocks[request_id] = set()
    
    def unregister_request(self, request_id: str) -> None:
        """Remove a completed request from tracking.
        
        Args:
            request_id: ID of the completed request
        """
        # Add to cumulative count before clearing
        if request_id in self.evicted_blocks:
            self.cumulative_evicted_blocks += len(self.evicted_blocks[request_id])
        
        self.request_blocks.pop(request_id, None)
        self.evicted_blocks.pop(request_id, None)
        self.l2norm_manager.clear_request(request_id)
    
    def update_blocks(
        self,
        request_id: str,
        block_ids: Dict[int, List[int]]
    ) -> None:
        """Update block tracking for a request.
        
        Args:
            request_id: ID of the request
            block_ids: Mapping of layer_idx -> block_ids
        """
        self.register_request(request_id)
        self.request_blocks[request_id] = block_ids
    
    def compute_l2norms_for_request(
        self,
        request_id: str,
        kv_caches: List[torch.Tensor],  # [(2, num_blocks, block_size, num_kv_heads, head_size), ...]
        block_ids: List[List[int]],  # [layer_idx: block_ids]
    ) -> None:
        """Compute and store L2-norms for a request's blocks.
        
        Args:
            request_id: ID of the request
            kv_caches: KV cache tensors per layer
            block_ids: Block IDs per layer
        """
        for layer_idx, (kv_cache, layer_block_ids) in enumerate(zip(kv_caches, block_ids)):
            if len(layer_block_ids) == 0:
                continue
                
            # Compute scores for this layer's blocks
            scores = compute_block_l2norms_from_cache(
                kv_cache,
                layer_block_ids,
                scoring_method=self.config.evict_method 
                    if self.config.evict_method in ["value_l2", "key_l2", "value_l2_div_key_l2", "value_l2_plus_key_l2"]
                    else "value_l2_div_key_l2"  # Default to paper's method
            )
            
            # Store scores
            self.l2norm_manager.update_block_l2norms(
                request_id,
                layer_idx,
                scores.tolist()
            )
    
    def should_evict(
        self,
        request_id: str,
        current_tokens: int
    ) -> bool:
        """Check if eviction should be triggered for a request.
        
        Args:
            request_id: ID of the request
            current_tokens: Current number of tokens in cache
            
        Returns:
            True if eviction should be triggered
        """
        if self.config.disable_evict_prefill:
            logger.debug(
                "[EVICTION DEBUG] should_evict: disabled (disable_evict_prefill=True)"
            )
            return False
        
        # Get the effective budget (dynamic if ratio is set)
        effective_budget = self.config.get_budget_for_tokens(current_tokens)
        
        result = should_trigger_eviction(
            current_tokens=current_tokens,
            cache_budget=effective_budget,
            block_size=self.block_size,
        )
        
        logger.debug(
            "[EVICTION DEBUG] should_evict: request=%s, current_tokens=%d, "
            "effective_budget=%d, block_size=%d, should_evict=%s",
            request_id, current_tokens, effective_budget, self.block_size, result
        )
        
        return result
    
    def get_blocks_to_evict(
        self,
        request_id: str,
        current_tokens: int,
        num_total_blocks: int,
        block_ids: List[int] | None = None,
    ) -> Set[int]:
        """Get the set of block IDs that should be evicted.
        
        Uses L2-norm scores to identify the lowest-scoring blocks
        for eviction. Strictly faithful to algorithms: checks ALL blocks.
        
        Args:
            request_id: ID of the request
            current_tokens: Current number of tokens
            num_total_blocks: Total number of blocks for the request
            block_ids: Optional list of block IDs for the request. If not provided,
                      falls back to cached request_blocks.
            
        Returns:
            Set of block IDs to evict
        """
        if not self.should_evict(request_id, current_tokens):
            return set()
        
        # Calculate how many blocks to evict using block-based budget
        target_blocks = self.config.get_budget_blocks(num_total_blocks, self.block_size)
        
        # Calculate resident blocks: Logical Total - Already Evicted
        num_previously_evicted = len(self.evicted_blocks.get(request_id, set()))
        num_resident_blocks = num_total_blocks - num_previously_evicted
        
        # We need to evict enough to get resident down to target
        num_blocks_to_evict = max(0, num_resident_blocks - target_blocks)
        
        if num_blocks_to_evict == 0:
            return set()
        
        # Get block IDs - prefer passed block_ids, fallback to cached
        all_block_ids = block_ids
        if all_block_ids is None:
            if request_id not in self.request_blocks:
                logger.debug(
                    "[EVICTION DEBUG] get_blocks_to_evict: request=%s not in request_blocks "
                    "and no block_ids passed, returning empty",
                    request_id
                )
                return set()
            layer_blocks = self.request_blocks[request_id]
            if not layer_blocks:
                return set()
            # Assuming block structure is consistent across layers, take first available
            first_layer_idx = next(iter(layer_blocks.keys()))
            all_block_ids = layer_blocks[first_layer_idx]
        
        # Evictable range: ALL blocks (0 to total), for strict algorithm faithfulness
        # We operate on the full logical list.
        start_idx = 0
        end_idx = num_total_blocks
        
        evictable_block_ids = all_block_ids[start_idx:end_idx]
        
        if not evictable_block_ids:
            return set()
            
        # Helper: Get scores for these blocks
        # We need to aggregate scores across layers for each block position
        # block_scores: List[Tuple[block_id, score]]
        block_scores = []
        
        # cached_l2_norms = self.l2norm_manager.get_block_l2norms(request_id, layer_idx)
        # We iterate over blocks by their LOGICAL index in the sequence (idx)
        # to fetch scores from L2NormManager which indexes by [layer][block_idx]
        
        # First, ensure we have data. If no L2 norms are recorded yet, we can't score.
        
        # We'll compute aggregate score for every block in the evictable range
        # Note: evictable_block_ids[i] corresponds to logical block index (start_idx + i)
        
        request_norms = self.l2norm_manager.reqid_mapping_block_l2norms.get(request_id) if self.l2norm_manager.reqid_mapping_block_l2norms else None
        
        if not request_norms:
             logger.debug(f"[EVICTION] No L2 norms found for {request_id}, skipping eviction this step.")
             return set()

        for local_i, block_id in enumerate(evictable_block_ids):
            # Skip already evicted
            if block_id in self.evicted_blocks.get(request_id, set()):
                continue
                
            logical_idx = start_idx + local_i
            
            # Aggregate score across all layers
            score_sum = 0.0
            valid_layers = 0
            
            for layer_idx, layer_norms in request_norms.items():
                if logical_idx < len(layer_norms):
                    score_sum += layer_norms[logical_idx]
                    valid_layers += 1
            
            if valid_layers > 0:
                avg_score = score_sum / valid_layers
                block_scores.append((block_id, avg_score))
            else:
                # If we have no info, maybe don't evict.
                pass

        if not block_scores:
            return set()

        # Sort by score ascending (lowest score = least important = evict first)
        block_scores.sort(key=lambda x: x[1])
        
        num_to_evict = min(num_blocks_to_evict, len(block_scores))
        blocks_to_evict = {block_id for block_id, _ in block_scores[:num_to_evict]}
        
        logger.debug(
            "[EVICTION] Evicting %d blocks for request %s (Score range: %.4f - %.4f)",
            len(blocks_to_evict), request_id, 
            block_scores[0][1] if block_scores else 0.0,
            block_scores[num_to_evict-1][1] if num_to_evict > 0 else 0.0
        )
        
        # Mark as evicted
        if request_id not in self.evicted_blocks:
            self.evicted_blocks[request_id] = set()
        self.evicted_blocks[request_id].update(blocks_to_evict)
        
        return blocks_to_evict
    
    def prune_prefill_kv(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prune KV tensors during prefill to meet budget constraint.
        
        Uses L2-norm scoring to identify and remove lowest-scoring tokens.
        
        Args:
            keys: Key tensor (seq_len, num_heads, head_dim)
            values: Value tensor (seq_len, num_heads, head_dim)
            
        Returns:
            Tuple of pruned (keys, values)
        """
        if self.config.disable_evict_prefill:
            return keys, values
        
        return self.pruner.prune_prompt(keys, values)
    
    def get_stats(self) -> Dict:
        """Get eviction statistics.
        
        Returns:
            Dictionary with eviction stats
        """
        # Current in-flight evictions
        current_evicted = sum(len(evicted) for evicted in self.evicted_blocks.values())
        # Total evictions including completed requests
        total_evicted = self.cumulative_evicted_blocks + current_evicted
        
        budget_info = (
            f"ratio={self.config.cache_budget_ratio}" 
            if self.config.cache_budget_ratio is not None 
            else f"budget={self.config.cache_budget}"
        )
        return {
            "num_tracked_requests": len(self.request_blocks),
            "total_evicted_blocks": total_evicted,
            "cumulative_evicted_blocks": self.cumulative_evicted_blocks,
            "current_request_evicted_blocks": current_evicted,
            "budget_mode": budget_info,
            "evict_method": self.config.evict_method,
        }
    
    def reset(self) -> None:
        """Reset all tracking state."""
        self.request_blocks.clear()
        self.evicted_blocks.clear()
        self.l2norm_manager.reset()
        self.cumulative_evicted_blocks = 0
