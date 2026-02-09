from pydantic import BaseModel
from typing import List, Optional, Tuple

class UpdateMaskRequest(BaseModel):
    request_id: str
    evictable_token_ranges: List[Tuple[int, int]]

class L2NormsRequest(BaseModel):
    request_id: str
    start_index: int = 0

class EvictKVBlocksRequest(BaseModel):
    request_id: str
    evictable_token_ranges: List[Tuple[int, int]]

class L2NormConfigRequest(BaseModel):
    """Request to configure L2 norm computation."""
    l2_norm_layers: Optional[List[int]] = None  # Layers to use (None = all)
    skip_layers: Optional[List[int]] = None     # Layers to skip
    enabled: Optional[bool] = None