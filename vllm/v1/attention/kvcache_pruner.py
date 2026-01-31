"""KV Cache Pruner for PagedEviction in vLLM v1 engine.

This module implements L2-norm based KV cache eviction following the paper's
methodology. The primary scoring method is `value_l2_div_key_l2` which computes
the ratio of value L2-norm to key L2-norm for each token.

Ported from vLLM-PagedEviction for v1 engine with FlashAttention backend.
"""

import math
import torch
import torch.nn.functional as F
from typing import Optional, Tuple


class KVCachePruner:
    """Prunes KV cache based on L2-norm scores.
    
    Supports multiple eviction methods:
    - value_l2_div_key_l2: Paper's method - ratio of value to key L2-norms (higher = keep)
    - global: Global eviction using configurable sub-method
    - local: Local eviction using configurable sub-method  
    - streamingLLM: Fixed window eviction (keep first and last blocks)
    - inverse_key_l2: Inverse of key L2-norm
    
    Args:
        cache_prune_type: Pruning type, currently only "budget" supported
        evict_method: Eviction method to use
        block_size: Number of tokens per KV cache block
        cache_budget: Maximum number of tokens to keep in cache
        initial_blocks: Number of initial blocks to always preserve (sink tokens)
    """
    
    def __init__(
        self, 
        cache_prune_type: str, 
        evict_method: str, 
        block_size: int, 
        cache_budget: int, 
        initial_blocks: int = 1
    ):
        self.cache_prune_type = cache_prune_type
        self.evict_method = evict_method
        self.orig_block_size = block_size
        
        assert self.cache_prune_type == "budget", \
            "Only 'budget' cache_prune_type is supported"
        
        self.cache_budget = cache_budget
        self.initial_blocks = initial_blocks
        
        # Sub-method for global/local eviction
        # Default to value_l2_div_key_l2 (paper's method)
        self.sub_evict_method = "value_l2_div_key_l2"
        
        assert self.cache_budget >= 3 * self.orig_block_size, \
            "Cache budget must be at least 3 times the block size"

    # =========================================================================
    # Scoring Methods - L2 Norm Based
    # =========================================================================
    
    def key_l1(self, key_block: torch.Tensor) -> torch.Tensor:
        """Compute L1 norm of key tensor along head dimension."""
        return torch.norm(key_block, p=1, dim=-1)
    
    def key_l2(self, key_block: torch.Tensor) -> torch.Tensor:
        """Compute L2 norm of key tensor along head dimension."""
        return torch.norm(key_block, p=2, dim=-1)
    
    def value_l1(self, value_block: torch.Tensor) -> torch.Tensor:
        """Compute L1 norm of value tensor along head dimension."""
        return torch.norm(value_block, p=1, dim=-1)
    
    def value_l2(self, value_block: torch.Tensor) -> torch.Tensor:
        """Compute L2 norm of value tensor along head dimension."""
        return torch.norm(value_block, p=2, dim=-1)
    
    def inverse_key_l1(self, key_block: torch.Tensor, value_block: torch.Tensor) -> torch.Tensor:
        """Compute inverse of key L1 norm."""
        return torch.div(1, self.key_l1(key_block) + 1e-8)
    
    def inverse_key_l2(self, key_block: torch.Tensor, value_block: torch.Tensor) -> torch.Tensor:
        """Compute inverse of key L2 norm."""
        return torch.div(1, self.key_l2(key_block) + 1e-8)
    
    # =========================================================================
    # Ratio-Based Scoring Methods
    # =========================================================================
    
    def key_l1_div_value_l1(self, key_block: torch.Tensor, value_block: torch.Tensor) -> torch.Tensor:
        """Compute key L1 / value L1 ratio."""
        return torch.div(self.key_l1(key_block), self.value_l1(value_block) + 1e-8)
    
    def key_l1_div_value_l2(self, key_block: torch.Tensor, value_block: torch.Tensor) -> torch.Tensor:
        """Compute key L1 / value L2 ratio."""
        return torch.div(self.key_l1(key_block), self.value_l2(value_block) + 1e-8)
    
    def key_l2_div_value_l1(self, key_block: torch.Tensor, value_block: torch.Tensor) -> torch.Tensor:
        """Compute key L2 / value L1 ratio."""
        return torch.div(self.key_l2(key_block), self.value_l1(value_block) + 1e-8)
    
    def key_l2_div_value_l2(self, key_block: torch.Tensor, value_block: torch.Tensor) -> torch.Tensor:
        """Compute key L2 / value L2 ratio."""
        return torch.div(self.key_l2(key_block), self.value_l2(value_block) + 1e-8)
    
    def value_l1_div_key_l1(self, key_block: torch.Tensor, value_block: torch.Tensor) -> torch.Tensor:
        """Compute value L1 / key L1 ratio."""
        return torch.div(self.value_l1(value_block), self.key_l1(key_block) + 1e-8)
    
    def value_l1_div_key_l2(self, key_block: torch.Tensor, value_block: torch.Tensor) -> torch.Tensor:
        """Compute value L1 / key L2 ratio."""
        return torch.div(self.value_l1(value_block), self.key_l2(key_block) + 1e-8)
    
    def value_l2_div_key_l1(self, key_block: torch.Tensor, value_block: torch.Tensor) -> torch.Tensor:
        """Compute value L2 / key L1 ratio."""
        return torch.div(self.value_l2(value_block), self.key_l1(key_block) + 1e-8)
    
    def value_l2_div_key_l2(self, key_block: torch.Tensor, value_block: torch.Tensor) -> torch.Tensor:
        """Compute value L2 / key L2 ratio - paper's method.
        
        This is the primary scoring method described in the PagedEviction paper.
        Higher scores indicate more important tokens that should be preserved.
        """
        return torch.div(self.value_l2(value_block), self.key_l2(key_block) + 1e-8)
    
    # =========================================================================
    # Sum-Based Scoring Methods
    # =========================================================================
    
    def value_l1_plus_key_l1(self, key_block: torch.Tensor, value_block: torch.Tensor) -> torch.Tensor:
        """Compute value L1 + key L1 sum."""
        return self.value_l1(value_block) + self.key_l1(key_block)
    
    def value_l2_plus_key_l2(self, key_block: torch.Tensor, value_block: torch.Tensor) -> torch.Tensor:
        """Compute value L2 + key L2 sum."""
        return self.value_l2(value_block) + self.key_l2(key_block)
    
    # =========================================================================
    # Score Dispatcher
    # =========================================================================
    
    def get_score(self, block_keys: torch.Tensor, block_values: torch.Tensor, evict_method: str) -> torch.Tensor:
        """Compute scores for tokens based on the specified eviction method.
        
        Args:
            block_keys: Key tensor of shape (num_tokens, num_heads, head_dim)
            block_values: Value tensor of shape (num_tokens, num_heads, head_dim)
            evict_method: Eviction scoring method to use
            
        Returns:
            Tensor of scores, shape (num_tokens, num_heads)
            Higher scores indicate more important tokens to keep.
        """
        if evict_method == "cosine":
            a_norm = F.normalize(block_keys, dim=-1)
            b_norm = F.normalize(block_values, dim=-1)
            return torch.sum(a_norm * b_norm, dim=-1)
        
        elif evict_method == "key_l1":
            return self.key_l1(block_keys)
        
        elif evict_method == "key_l2":
            return self.key_l2(block_keys)
        
        elif evict_method == "value_l1":
            return self.value_l1(block_values)
        
        elif evict_method == "value_l2":
            return self.value_l2(block_values)
        
        elif evict_method == "key_l1_div_value_l1":
            return self.key_l1_div_value_l1(block_keys, block_values)
        
        elif evict_method == "key_l1_div_value_l2":
            return self.key_l1_div_value_l2(block_keys, block_values)
        
        elif evict_method == "key_l2_div_value_l1":
            return self.key_l2_div_value_l1(block_keys, block_values)
        
        elif evict_method == "key_l2_div_value_l2":
            return self.key_l2_div_value_l2(block_keys, block_values)
        
        elif evict_method == "value_l1_div_key_l1":
            return self.value_l1_div_key_l1(block_keys, block_values)
        
        elif evict_method == "value_l1_div_key_l2":
            return self.value_l1_div_key_l2(block_keys, block_values)
        
        elif evict_method == "value_l2_div_key_l1":
            return self.value_l2_div_key_l1(block_keys, block_values)
        
        elif evict_method == "value_l2_div_key_l2":
            # Paper's primary method
            return self.value_l2_div_key_l2(block_keys, block_values)
        
        elif evict_method == "value_l1_plus_key_l1":
            return self.value_l1_plus_key_l1(block_keys, block_values)
        
        elif evict_method == "value_l2_plus_key_l2":
            return self.value_l2_plus_key_l2(block_keys, block_values)
        
        elif evict_method == "streamingLLM":
            # StreamingLLM: score by position (higher position = higher score = keep recent)
            q_len, num_heads, head_dim = block_keys.shape
            return torch.arange(1, q_len + 1, device=block_keys.device).view(-1, 1).expand(-1, num_heads).float()
        
        elif evict_method == "inverse_key_l1":
            return self.inverse_key_l1(block_keys, block_values)
        
        elif evict_method == "inverse_key_l2":
            return self.inverse_key_l2(block_keys, block_values)
        
        else:
            raise ValueError(f"Unknown eviction method: {evict_method}")
    
    def get_token_score(self, keys: torch.Tensor, values: torch.Tensor, evict_method: str) -> torch.Tensor:
        """Compute per-token scores for global/local eviction.
        
        Used by global and local eviction methods which operate on individual tokens.
        
        Args:
            keys: Key tensor of shape (num_tokens, num_heads, head_dim)
            values: Value tensor of shape (num_tokens, num_heads, head_dim)
            evict_method: High-level eviction method (global/local/etc)
            
        Returns:
            Tensor of scores, shape (num_tokens, num_heads)
        """
        if evict_method in ['global', 'local']:
            # Use sub_evict_method for actual scoring
            return self.get_score(keys, values, self.sub_evict_method)
        elif evict_method == "inverse_key_l2":
            return self.inverse_key_l2(keys, values)
        else:
            # Fall back to direct method
            return self.get_score(keys, values, evict_method)
    
    def get_block_score(self, block_keys: torch.Tensor, block_values: torch.Tensor, evict_method: str) -> torch.Tensor:
        """Compute aggregate score for a block of tokens.
        
        Used for block-level eviction decisions in global/local methods.
        
        Args:
            block_keys: Key tensor of shape (block_size, num_heads, head_dim)
            block_values: Value tensor of shape (block_size, num_heads, head_dim)
            evict_method: Must be 'global' or 'local'
            
        Returns:
            Scalar tensor representing aggregate block score.
        """
        assert evict_method in ["global", "local"], \
            "Only global and local eviction methods support block score calculation"
        
        if self.sub_evict_method == "value_l2":
            return torch.norm(block_values, p=2, dim=-1).mean(dim=1).sum(dim=0)
        elif self.sub_evict_method == "key_l2":
            return torch.norm(block_keys, p=2, dim=-1).mean(dim=1).sum(dim=0)
        elif self.sub_evict_method == "value_l2_plus_key_l2":
            value_norms = torch.norm(block_values, p=2, dim=-1).mean(dim=1).sum(dim=0)
            key_norms = torch.norm(block_keys, p=2, dim=-1).mean(dim=1).sum(dim=0)
            return value_norms + key_norms
        elif self.sub_evict_method == "inverse_key_l2":
            key_norms = torch.norm(block_keys, p=2, dim=-1).mean(dim=1).sum(dim=0)
            return 1 / (key_norms + 1e-8)
        elif self.sub_evict_method == "value_l2_div_key_l2":
            # Paper's method for block scoring
            value_norms = torch.norm(block_values, p=2, dim=-1).mean(dim=1).sum(dim=0)
            key_norms = torch.norm(block_keys, p=2, dim=-1).mean(dim=1).sum(dim=0)
            return value_norms / (key_norms + 1e-8)
        else:
            raise ValueError(f"Unknown sub_evict_method: {self.sub_evict_method}")

    # =========================================================================
    # Prompt Pruning (Prefill-time eviction)
    # =========================================================================
    
    def prune_prompt(
        self, 
        key_tensor: torch.Tensor, 
        value_tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prune KV cache during prefill to fit within cache budget.
        
        Keeps the first block (sink tokens) and last block (recent context),
        and prunes the middle based on L2-norm scores.
        
        Args:
            key_tensor: Key tensor of shape (seq_len, num_heads, head_dim)
            value_tensor: Value tensor of shape (seq_len, num_heads, head_dim)
            
        Returns:
            Tuple of (pruned_keys, pruned_values) tensors
        """
        q_len, num_heads, head_dim = key_tensor.shape
        
        # No pruning needed if under budget
        if q_len <= self.cache_budget:
            return key_tensor, value_tensor
        
        if self.evict_method in ["streamingLLM", "streamingLLM-1"]:
            return self._prune_prompt_streaming(key_tensor, value_tensor, q_len, num_heads, head_dim)
        elif self.evict_method in ["local", "global", "inverse_key_l2", "value_l2_div_key_l2"]:
            return self._prune_prompt_scored(key_tensor, value_tensor, q_len, num_heads, head_dim)
        else:
            raise ValueError(f"Unknown evict_method: {self.evict_method}")
    
    def _prune_prompt_streaming(
        self,
        key_tensor: torch.Tensor,
        value_tensor: torch.Tensor,
        q_len: int,
        num_heads: int,
        head_dim: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """StreamingLLM style pruning - drop oldest tokens from middle."""
        remainder_size = q_len % self.orig_block_size
        
        # First block slice (sink tokens)
        end_idx_first_slice = self.orig_block_size
        first_slice = slice(0, end_idx_first_slice)
        
        # Middle slice (to be pruned)
        end_idx_middle_slice = q_len - self.orig_block_size - remainder_size
        
        # Last slice (recent context)
        last_slice = slice(end_idx_middle_slice, q_len)
        
        # Calculate pruning
        middle_tokens = end_idx_middle_slice - end_idx_first_slice
        middle_pruned_tokens = self.cache_budget - self.orig_block_size - self.orig_block_size
        total_prune_tokens = max(0, middle_tokens - middle_pruned_tokens)
        
        # Middle slice after pruning (drop oldest)
        middle_slice = slice(end_idx_first_slice + total_prune_tokens, end_idx_middle_slice)
        
        # Rejoin slices
        rejoined_key = torch.cat([
            key_tensor[first_slice],
            key_tensor[middle_slice],
            key_tensor[last_slice]
        ], dim=0)
        
        rejoined_value = torch.cat([
            value_tensor[first_slice],
            value_tensor[middle_slice],
            value_tensor[last_slice]
        ], dim=0)
        
        return rejoined_key, rejoined_value
    
    def _prune_prompt_scored(
        self,
        key_tensor: torch.Tensor,
        value_tensor: torch.Tensor,
        q_len: int,
        num_heads: int,
        head_dim: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Score-based pruning - drop lowest scoring tokens from middle."""
        remainder_size = q_len % self.orig_block_size
        
        # First block slice (always keep)
        end_idx_first_slice = self.orig_block_size
        first_slice = slice(0, end_idx_first_slice)
        
        # Middle slice (to be pruned)
        end_idx_middle_slice = q_len - self.orig_block_size - remainder_size
        middle_slice = slice(end_idx_first_slice, end_idx_middle_slice)
        
        # Last slice (always keep)
        last_slice = slice(end_idx_middle_slice, q_len)
        
        # Extract middle tokens
        middle_key = key_tensor[middle_slice]
        middle_value = value_tensor[middle_slice]
        middle_tokens = middle_key.shape[0]
        
        # Calculate how many to prune
        middle_pruned_tokens = self.cache_budget - self.orig_block_size - self.orig_block_size
        total_prune_tokens = max(0, middle_tokens - middle_pruned_tokens)
        
        if total_prune_tokens == 0:
            return key_tensor, value_tensor
        
        # Get scores for middle tokens
        scores = self.get_token_score(middle_key, middle_value, self.evict_method)
        
        # Find tokens with lowest scores (to evict)
        _, least_indices = torch.topk(scores, k=total_prune_tokens, largest=False, dim=0)
        
        # Create mask (True = keep, False = evict)
        mask = torch.ones_like(scores, dtype=torch.bool)
        mask.scatter_(0, least_indices, False)
        
        # Apply mask to prune middle tokens
        pruned_middle_key = middle_key[mask].view(-1, num_heads, head_dim)
        pruned_middle_value = middle_value[mask].view(-1, num_heads, head_dim)
        
        # Rejoin slices
        rejoined_key = torch.cat([
            key_tensor[first_slice],
            pruned_middle_key,
            key_tensor[last_slice]
        ], dim=0)
        
        rejoined_value = torch.cat([
            value_tensor[first_slice],
            pruned_middle_value,
            value_tensor[last_slice]
        ], dim=0)
        
        return rejoined_key, rejoined_value
    
    def get_pruned_length(self, original_length: int) -> int:
        """Calculate the length after pruning.
        
        Args:
            original_length: Original sequence length
            
        Returns:
            Sequence length after pruning (or original if under budget)
        """
        if original_length <= self.cache_budget:
            return original_length
        return min(original_length, self.cache_budget)

    # =========================================================================
    # Decode-time Block Eviction
    # =========================================================================
    
    def get_blocks_to_prune_decode(self, seq_kv_len: int) -> Tuple[int, int, int]:
        """Determine which blocks to prune during decode.
        
        For budget-based eviction, evicts oldest middle block when over budget.
        
        Args:
            seq_kv_len: Current KV cache sequence length
            
        Returns:
            Tuple of (start_block_id, end_block_id, pruned_tokens)
            Returns (-1, -1, 0) if no pruning needed.
        """
        if self.evict_method in ["streamingLLM", "streamingLLM-1"]:
            if seq_kv_len <= self.cache_budget:
                return -1, -1, 0
            else:
                # Evict one block after initial_blocks
                s_block_id = self.initial_blocks
                e_block_id = self.initial_blocks + 1
                prune_tokens = self.orig_block_size
                return s_block_id, e_block_id, prune_tokens
        else:
            # For global/local/inverse_key_l2/value_l2_div_key_l2
            if seq_kv_len <= self.cache_budget:
                return -1, -1, 0
            else:
                # Return block range for scoring-based eviction
                s_block_id = self.initial_blocks
                e_block_id = self.initial_blocks + 1
                prune_tokens = self.orig_block_size
                return s_block_id, e_block_id, prune_tokens
    
    def prune_oldest_block(
        self, 
        key_tensor: torch.Tensor, 
        value_tensor: torch.Tensor, 
        prune_tokens: int
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Prune tokens from oldest middle block during decode.
        
        For streamingLLM, simply removes oldest block.
        For score-based methods, evicts lowest-scoring tokens.
        
        Args:
            key_tensor: Key tensor for the block(s) to prune
            value_tensor: Value tensor for the block(s) to prune
            prune_tokens: Number of tokens to prune
            
        Returns:
            Tuple of (pruned_keys, pruned_values) or (None, None) for streamingLLM
        """
        if self.evict_method in ["streamingLLM", "streamingLLM-1"]:
            # StreamingLLM just drops the block, handled by block manager
            return None, None
        
        q_len, num_heads, head_dim = key_tensor.shape
        
        # Get scores
        scores = self.get_score(key_tensor, value_tensor, self.evict_method)
        
        # Find lowest scoring tokens
        _, least_indices = torch.topk(scores, k=prune_tokens, largest=False, dim=0)
        
        # Create mask
        mask = torch.ones_like(scores, dtype=torch.bool)
        mask.scatter_(0, least_indices, False)
        
        # Prune tokens
        pruned_key = key_tensor[mask].view(-1, num_heads, head_dim)
        pruned_value = value_tensor[mask].view(-1, num_heads, head_dim)
        
        return pruned_key, pruned_value


# =============================================================================
# Utility Functions
# =============================================================================

def compute_block_l2norms_from_cache(
    kv_cache: torch.Tensor,
    block_ids: list[int],
    scoring_method: str = "value_l2_div_key_l2"
) -> torch.Tensor:
    """Compute L2-norm scores for specified blocks from KV cache.
    
    This is used with FlashAttention backend where we compute scores
    from the cached KV values rather than during attention computation.
    
    Args:
        kv_cache: KV cache tensor [2, num_blocks, block_size, num_kv_heads, head_size]
        block_ids: List of block IDs to score
        scoring_method: Method for computing scores
        
    Returns:
        Tensor of shape (len(block_ids),) with aggregate score per block.
    """
    if len(block_ids) == 0:
        return torch.tensor([], device=kv_cache.device)
    
    key_cache = kv_cache[0]  # [num_blocks, block_size, num_kv_heads, head_size]
    value_cache = kv_cache[1]
    
    # Get specified blocks
    k_blocks = key_cache[block_ids]  # [len(block_ids), block_size, num_kv_heads, head_size]
    v_blocks = value_cache[block_ids]
    
    if scoring_method == "value_l2":
        # Compute L2 norms of values
        v_norms = torch.norm(v_blocks, p=2, dim=-1)  # [..., block_size, num_kv_heads]
        block_scores = v_norms.mean(dim=-1).sum(dim=-1)  # [len(block_ids)]
    elif scoring_method == "key_l2":
        k_norms = torch.norm(k_blocks, p=2, dim=-1)
        block_scores = k_norms.mean(dim=-1).sum(dim=-1)
    elif scoring_method == "value_l2_div_key_l2":
        # Paper's method
        v_norms = torch.norm(v_blocks, p=2, dim=-1).mean(dim=-1).sum(dim=-1)
        k_norms = torch.norm(k_blocks, p=2, dim=-1).mean(dim=-1).sum(dim=-1)
        block_scores = v_norms / (k_norms + 1e-8)
    elif scoring_method == "value_l2_plus_key_l2":
        v_norms = torch.norm(v_blocks, p=2, dim=-1).mean(dim=-1).sum(dim=-1)
        k_norms = torch.norm(k_blocks, p=2, dim=-1).mean(dim=-1).sum(dim=-1)
        block_scores = v_norms + k_norms
    else:
        raise ValueError(f"Unknown scoring method: {scoring_method}")
    
    return block_scores
