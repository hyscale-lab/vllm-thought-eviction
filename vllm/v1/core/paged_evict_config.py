"""PagedEviction configuration for vLLM v1 engine.

Defines configuration for L2-norm based KV cache eviction.
Ported from vLLM-PagedEviction.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PagedEvictConfig:
    """Configuration for PagedEviction KV cache management.
    
    PagedEviction uses L2-norm based scoring to determine which KV cache
    blocks to evict when memory budget is exceeded.
    
    Attributes:
        disable_evict_prefill: If True, skip eviction during prefill phase
        cache_prune_type: Type of pruning, currently only "budget" supported
        evict_method: Scoring method for eviction:
            - "value_l2_div_key_l2": Paper's method - ratio of value to key L2-norm
            - "global": Global eviction with configurable sub-method
            - "local": Local eviction with configurable sub-method
            - "streamingLLM": Fixed window eviction
            - "inverse_key_l2": Inverse of key L2-norm
        cache_budget: Maximum number of tokens to keep in KV cache per sequence.
            If None, cache_budget_ratio is used instead.
        cache_budget_ratio: Ratio of tokens to retain (0.0-1.0). When set,
            the budget is calculated dynamically as current_tokens * ratio.
            Useful for streaming where context grows over time.
        topk_blocks: Number of top blocks to consider for eviction (-1 = all)
        initial_blocks: Number of initial blocks to always preserve (sink tokens)
    """
    
    disable_evict_prefill: bool = False
    cache_prune_type: str = "budget"
    evict_method: str = "value_l2_div_key_l2"
    cache_budget: int | None = 512
    cache_budget_ratio: float | None = None  # e.g., 0.5 = keep 50% of tokens
    topk_blocks: int = -1
    initial_blocks: int = 1
    sampled_layers: list[int] | None = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._verify_args()
    
    def _verify_args(self) -> None:
        """Validate configuration arguments."""
        # Validate cache_prune_type
        if self.cache_prune_type not in ["percentage", "budget"]:
            raise ValueError(
                f"cache_prune_type must be 'percentage' or 'budget', "
                f"got '{self.cache_prune_type}'"
            )
        
        # Currently only budget is supported
        assert self.cache_prune_type == "budget", \
            "Cache prune type must be 'budget' for paged compression"
        
        # Validate evict_method
        valid_methods = [
            "value_l2_div_key_l2",  # Paper's method
            "global", 
            "local", 
            "streamingLLM", 
            "streamingLLM-1",
            "inverse_key_l2",
            "value_l2",
            "key_l2",
            "value_l2_plus_key_l2",
        ]
        
        if self.evict_method not in valid_methods:
            raise ValueError(
                f"evict_method must be one of {valid_methods}, "
                f"got '{self.evict_method}'"
            )
        
        # Validate cache_budget and cache_budget_ratio
        if self.cache_budget is None and self.cache_budget_ratio is None:
            raise ValueError(
                "Either cache_budget or cache_budget_ratio must be set"
            )
        
        if self.cache_budget is not None and self.cache_budget <= 0:
            raise ValueError(
                f"cache_budget must be positive, got {self.cache_budget}"
            )
        
        if self.cache_budget_ratio is not None:
            if not 0.0 < self.cache_budget_ratio <= 1.0:
                raise ValueError(
                    f"cache_budget_ratio must be in (0.0, 1.0], got {self.cache_budget_ratio}"
                )
        
        # Validate topk_blocks
        if self.topk_blocks != -1 and self.topk_blocks <= 0:
            raise ValueError(
                f"topk_blocks must be -1 or positive, got {self.topk_blocks}"
            )
        
        # Validate initial_blocks
        if self.initial_blocks < 0:
            raise ValueError(
                f"initial_blocks must be non-negative, got {self.initial_blocks}"
            )

        # Validate sampled_layers
        if self.sampled_layers is not None:
             if not isinstance(self.sampled_layers, list):
                 raise ValueError(f"sampled_layers must be a list of integers, got {type(self.sampled_layers)}")
             if any(not isinstance(i, int) for i in self.sampled_layers):
                 raise ValueError("sampled_layers must be a list of integers")
             if any(i < 0 for i in self.sampled_layers):
                 raise ValueError("sampled_layers indices must be non-negative")
    
    def get_budget_blocks(self, num_blocks: int, block_size: int = 16) -> int:
        """Calculate the effective budget in blocks based on current block count.
        
        If cache_budget_ratio is set, dynamically calculates budget as a
        percentage of current blocks. Otherwise, converts fixed token budget
        to blocks.
        
        Args:
            num_blocks: Current number of blocks in cache.
            block_size: Block size in tokens (default 16).
            
        Returns:
            Number of blocks to keep (budget in blocks).
        """
        if self.cache_budget_ratio is not None:
            # Percentage-based: keep ratio * num_blocks
            return max(1, int(num_blocks * self.cache_budget_ratio))
        else:
            # Fixed budget: convert tokens to blocks
            budget_tokens = self.cache_budget or 512
            return (budget_tokens + block_size - 1) // block_size
    
    def get_budget_for_tokens(self, current_tokens: int, block_size: int = 16) -> int:
        """Calculate the effective token budget based on current token count.
        
        Wrapper that converts block budget back to tokens for compatibility.
        
        Args:
            current_tokens: Current number of tokens in context.
            block_size: Block size in tokens (default 16).
            
        Returns:
            Effective token budget for eviction decisions.
        """
        num_blocks = (current_tokens + block_size - 1) // block_size
        budget_blocks = self.get_budget_blocks(num_blocks, block_size)
        return budget_blocks * block_size
    
    def __repr__(self) -> str:
        budget_info = (
            f"cache_budget_ratio={self.cache_budget_ratio}" 
            if self.cache_budget_ratio is not None 
            else f"cache_budget={self.cache_budget}"
        )
        return (
            f"PagedEvictConfig("
            f"disable_evict_prefill={self.disable_evict_prefill}, "
            f"cache_prune_type='{self.cache_prune_type}', "
            f"evict_method='{self.evict_method}', "
            f"{budget_info}, "
            f"topk_blocks={self.topk_blocks}, "
            f"initial_blocks={self.initial_blocks}, "
            f"sampled_layers={self.sampled_layers})"
        )
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "PagedEvictConfig":
        """Create config from dictionary."""
        return cls(
            disable_evict_prefill=config_dict.get("disable_evict_prefill", False),
            cache_prune_type=config_dict.get("cache_prune_type", "budget"),
            evict_method=config_dict.get("evict_method", "value_l2_div_key_l2"),
            cache_budget=config_dict.get("cache_budget", 512),
            cache_budget_ratio=config_dict.get("cache_budget_ratio"),
            topk_blocks=config_dict.get("topk_blocks", -1),
            initial_blocks=config_dict.get("initial_blocks", 1),
            sampled_layers=config_dict.get("sampled_layers", None),
        )
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            "disable_evict_prefill": self.disable_evict_prefill,
            "cache_prune_type": self.cache_prune_type,
            "evict_method": self.evict_method,
            "cache_budget": self.cache_budget,
            "cache_budget_ratio": self.cache_budget_ratio,
            "topk_blocks": self.topk_blocks,
            "initial_blocks": self.initial_blocks,
            "sampled_layers": self.sampled_layers,
        }
