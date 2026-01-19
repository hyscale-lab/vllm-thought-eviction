from http import HTTPStatus
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse

from vllm.entrypoints.openai.extensions.protocol import (
    UpdateMaskRequest, L2NormsRequest, 
    EvictKVBlocksRequest, L2NormConfigRequest
)

from vllm.entrypoints.openai.api_server import engine_client
from vllm.logger import init_logger


logger = init_logger(__name__)

router = APIRouter()

@router.post("/v1/attention/l2_norms")
async def get_l2_norms(request: L2NormsRequest, raw_request: Request):
    """
    Endpoint to get L2 norms of attention keys for a running request.
    Used for L2 norm-based KV cache eviction decisions.
    
    Returns:
        JSON with 'l2_norms' containing per-token L2 norms, or None if unavailable.
    """
    engine = engine_client(raw_request)

    # Check if the engine supports L2 norm retrieval
    if hasattr(engine, "get_request_l2_norms"):
        l2_norms = await engine.get_request_l2_norms(request.request_id, request.start_index)
        if l2_norms is not None:
            return JSONResponse({
                "success": True,
                "request_id": request.request_id,
                "l2_norms": l2_norms.tolist() if hasattr(l2_norms, 'tolist') else list(l2_norms)
            })
        else:
            return JSONResponse({
                "success": True,
                "request_id": request.request_id,
                "l2_norms": None,
                "message": "L2 norms not yet available for this request"
            })
    else:
        # Return empty response instead of error for compatibility
        return JSONResponse({
            "success": True,
            "request_id": request.request_id,
            "l2_norms": None,
            "message": "L2 norm retrieval not supported by current engine"
        })

@router.post("/v1/attention/update_mask")
async def update_attention_mask(request: UpdateMaskRequest,
                                raw_request: Request):
    """
    Endpoint to update the attention mask for a running request.
    Used for real-time KV cache eviction with FlexAttention.
    """
    engine = engine_client(raw_request)
    logger.info(request.request_id)
    logger.info(request.evictable_token_ranges)
    # The engine must be an AsyncLLM (V1) instance to have this method
    if hasattr(engine, "update_request_mask"):
        await engine.update_request_mask(request.request_id,
                                         request.evictable_token_ranges)
        return JSONResponse({"success": True})
    else:
        raise HTTPException(
            status_code=HTTPStatus.NOT_IMPLEMENTED,
            detail="The current engine does not support attention mask updates."
        )

@router.post("/v1/kv_cache/evict")
async def evict_kv_blocks(request: EvictKVBlocksRequest,
                          raw_request: Request):
    """
    Endpoint to trigger physical eviction of KV cache blocks.
    """
    engine = engine_client(raw_request)
    logger.info(f"Eviction request for {request.request_id} with {len(request.evictable_token_ranges)} ranges")
    
    # Check for support in AsyncLLM
    if hasattr(engine, "evict_kv_blocks"):
        await engine.evict_kv_blocks(request.request_id,
                                     request.evictable_token_ranges)
        return JSONResponse({"success": True})
    # Fallback to update_request_mask if evict_kv_blocks not explicit but update_request_mask is
    elif hasattr(engine, "update_request_mask"):
        await engine.update_request_mask(request.request_id,
                                         request.evictable_token_ranges)
        return JSONResponse({"success": True})
    else:
        raise HTTPException(
            status_code=HTTPStatus.NOT_IMPLEMENTED,
            detail="The current engine does not support KV cache eviction."
        )
        

@router.post("/v1/attention/l2_norms/config")
async def configure_l2_norms(request: L2NormConfigRequest, raw_request: Request):
    """
    Endpoint to configure L2 norm computation for attention layers.
    
    Allows specifying which layers to use for L2 norm computation,
    similar to skip_layers in l2_compress.py.
    
    Args:
        l2_norm_layers: List of layer indices to use for L2 norm computation.
                        If None, all layers are used.
        skip_layers: List of layer indices to skip for L2 norm computation.
                     Takes precedence over l2_norm_layers.
        enabled: Enable or disable L2 norm computation globally.
    
    Returns:
        JSON with current configuration.
    """
    engine = engine_client(raw_request)
    
    # Check if engine supports RPC-based L2 norm configuration
    if hasattr(engine, 'engine_core') and hasattr(engine.engine_core, 'configure_l2_norms_async'):
        try:
            result = await engine.engine_core.configure_l2_norms_async(
                l2_norm_layers=request.l2_norm_layers,
                skip_layers=request.skip_layers,
                enabled=request.enabled if request.enabled is not None else True
            )
            
            if "error" in result:
                return JSONResponse({
                    "success": False,
                    "error": result["error"]
                }, status_code=500)
            
            return JSONResponse({
                "success": True,
                "config": result
            })
        except Exception as e:
            return JSONResponse({
                "success": False,
                "error": str(e)
            }, status_code=500)
    else:
        return JSONResponse({
            "success": False,
            "error": "L2 norm configuration not supported by current engine"
        }, status_code=501)


@router.get("/v1/attention/l2_norms/config")
async def get_l2_norms_config(raw_request: Request):
    """
    Endpoint to get current L2 norm configuration.
    
    Returns:
        JSON with current configuration including enabled state and layer filters.
    """
    engine = engine_client(raw_request)
    
    # Check if engine supports RPC-based L2 norm configuration
    if hasattr(engine, 'engine_core') and hasattr(engine.engine_core, 'configure_l2_norms_async'):
        try:
            # Get current config by calling with no changes
            result = await engine.engine_core.configure_l2_norms_async(
                l2_norm_layers=None,
                skip_layers=None,
                enabled=True  # No change - just read current state
            )
            
            if "error" in result:
                return JSONResponse({
                    "success": False,
                    "error": result["error"]
                }, status_code=500)
            
            return JSONResponse({
                "success": True,
                "config": result
            })
        except Exception as e:
            return JSONResponse({
                "success": False,
                "error": str(e)
            }, status_code=500)
    else:
        return JSONResponse({
            "success": False,
            "error": "L2 norm configuration not supported by current engine"
        }, status_code=501)