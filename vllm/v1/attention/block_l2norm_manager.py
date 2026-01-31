"""Block L2 Norm Manager for PagedEviction in vLLM v1 engine.

This module manages L2-norm tracking per request and per block for
score-based KV cache eviction.

Ported from vLLM-PagedEviction for v1 engine with FlashAttention backend.
"""

from typing import Dict, List, Optional
import torch


class BlockL2NormManager:
    """Manages L2-norm scores for KV cache blocks across requests.
    
    Tracks block tables and their corresponding L2-norm scores for each
    active request. Used to make informed eviction decisions based on
    block importance scores.
    
    The manager maintains two primary mappings:
    - reqid_mapping_blocktables: Maps request_id -> layer_idx -> block_ids
    - reqid_mapping_block_l2norms: Maps request_id -> layer_idx -> l2_norms
    """
    
    def __init__(self):
        # Maps request_id to {layer_idx: [block_ids]}
        self.reqid_mapping_blocktables: Optional[Dict[str, Dict[int, List[int]]]] = None
        
        # Maps request_id to {layer_idx: [l2_norms per block]}
        self.reqid_mapping_block_l2norms: Optional[Dict[str, Dict[int, List[float]]]] = None
    
    def update_reqs(self, reqid_mapping_blocktables: Dict[str, Dict[int, List[int]]]) -> None:
        """Update active requests and their block tables.
        
        Called when the set of active requests or their block tables change.
        Creates new entries for new requests and removes entries for
        completed requests.
        
        Args:
            reqid_mapping_blocktables: New mapping of request_id -> layer_idx -> block_ids
        """
        if self.reqid_mapping_blocktables is None:
            # First initialization
            self.reqid_mapping_blocktables = reqid_mapping_blocktables
            self.reqid_mapping_block_l2norms = {}
            for req_id in reqid_mapping_blocktables:
                self.reqid_mapping_block_l2norms[req_id] = {}
            return
        
        # Find new requests
        new_req_ids = set(reqid_mapping_blocktables.keys()) - set(self.reqid_mapping_blocktables.keys())
        
        # Find removed requests
        removed_req_ids = set(self.reqid_mapping_blocktables.keys()) - set(reqid_mapping_blocktables.keys())
        
        # Update mappings
        self.reqid_mapping_blocktables = reqid_mapping_blocktables
        
        # Add entries for new requests
        for req_id in new_req_ids:
            self.reqid_mapping_block_l2norms[req_id] = {}
        
        # Remove entries for completed requests
        for req_id in removed_req_ids:
            if req_id in self.reqid_mapping_block_l2norms:
                del self.reqid_mapping_block_l2norms[req_id]
    
    def update_block_l2norms(
        self, 
        request_id: str,
        layer_idx: int, 
        l2_norms: List[float]
    ) -> None:
        """Update L2-norm scores for a request's blocks at a specific layer.
        
        Args:
            request_id: ID of the request
            layer_idx: Layer index (0-indexed)
            l2_norms: List of L2-norm scores, one per block
        """
        if self.reqid_mapping_block_l2norms is None:
            self.reqid_mapping_block_l2norms = {}
        
        if request_id not in self.reqid_mapping_block_l2norms:
            self.reqid_mapping_block_l2norms[request_id] = {}
        
        self.reqid_mapping_block_l2norms[request_id][layer_idx] = l2_norms
    
    def get_block_l2norms(
        self, 
        request_id: str, 
        layer_idx: int
    ) -> Optional[List[float]]:
        """Get L2-norm scores for a request's blocks at a specific layer.
        
        Args:
            request_id: ID of the request
            layer_idx: Layer index
            
        Returns:
            List of L2-norm scores or None if not available
        """
        if self.reqid_mapping_block_l2norms is None:
            return None
        
        if request_id not in self.reqid_mapping_block_l2norms:
            return None
        
        return self.reqid_mapping_block_l2norms[request_id].get(layer_idx)
    
    def get_lowest_scoring_block(
        self, 
        request_id: str,
        initial_blocks: int = 1
    ) -> Optional[int]:
        """Get the block index with the lowest aggregate score across layers.
        
        Excludes initial_blocks sink blocks and the last block from eviction.
        
        Args:
            request_id: ID of the request
            initial_blocks: Number of initial blocks to preserve
            
        Returns:
            Block index to evict, or None if no evictable blocks
        """
        if self.reqid_mapping_block_l2norms is None:
            return None
        
        if request_id not in self.reqid_mapping_block_l2norms:
            return None
        
        layer_norms = self.reqid_mapping_block_l2norms[request_id]
        if not layer_norms:
            return None
        
        # Aggregate scores across layers
        # Get any layer to find number of blocks
        first_layer_idx = next(iter(layer_norms.keys()))
        num_blocks = len(layer_norms[first_layer_idx])
        
        if num_blocks <= initial_blocks + 1:
            # Not enough blocks to evict
            return None
        
        # Compute aggregate score per block (excluding first and last)
        aggregate_scores = []
        for block_idx in range(initial_blocks, num_blocks - 1):
            total_score = 0.0
            for layer_idx, norms in layer_norms.items():
                if block_idx < len(norms):
                    total_score += norms[block_idx]
            aggregate_scores.append((block_idx, total_score))
        
        if not aggregate_scores:
            return None
        
        # Return block with lowest score
        lowest_block_idx = min(aggregate_scores, key=lambda x: x[1])[0]
        return lowest_block_idx
    
    def get_updated_block_tables(
        self,
        request_id: str,
        removed_block_idx: int
    ) -> Optional[Dict[int, List[int]]]:
        """Get updated block tables after removing a block.
        
        Args:
            request_id: ID of the request
            removed_block_idx: Index of the block to remove
            
        Returns:
            Updated block tables mapping, or None if request not found
        """
        if self.reqid_mapping_blocktables is None:
            return None
        
        if request_id not in self.reqid_mapping_blocktables:
            return None
        
        updated_tables = {}
        for layer_idx, block_ids in self.reqid_mapping_blocktables[request_id].items():
            # Remove the block at removed_block_idx
            updated_tables[layer_idx] = [
                bid for i, bid in enumerate(block_ids) 
                if i != removed_block_idx
            ]
        
        return updated_tables
    
    def clear_request(self, request_id: str) -> None:
        """Clear tracking data for a completed request.
        
        Args:
            request_id: ID of the request to clear
        """
        if self.reqid_mapping_blocktables is not None:
            self.reqid_mapping_blocktables.pop(request_id, None)
        
        if self.reqid_mapping_block_l2norms is not None:
            self.reqid_mapping_block_l2norms.pop(request_id, None)
    
    def reset(self) -> None:
        """Reset all tracking data."""
        self.reqid_mapping_blocktables = None
        self.reqid_mapping_block_l2norms = None
    
    @property
    def num_active_requests(self) -> int:
        """Get number of active requests being tracked."""
        if self.reqid_mapping_blocktables is None:
            return 0
        return len(self.reqid_mapping_blocktables)
