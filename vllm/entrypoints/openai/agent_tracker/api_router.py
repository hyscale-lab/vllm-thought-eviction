"""FastAPI router for the live trajectory tracker (D-16, D-18).

Routes:
  GET    /v1/agent_tracker/sessions/{session_id}/opportunity
  DELETE /v1/agent_tracker/sessions/{session_id}

Both look up the registry on `request.app.state.session_tracker_registry`,
which is stashed by `init_app_state` in `vllm/entrypoints/openai/api_server.py`.
"""
from __future__ import annotations

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse

from vllm.logger import init_logger

logger = init_logger(__name__)
router = APIRouter()


def _registry(request: Request):
    """Return the SessionTrackerRegistry stashed on app.state, or fallback
    to the module-level singleton (e.g., in unit tests without init_app_state)."""
    reg = getattr(request.app.state, "session_tracker_registry", None)
    if reg is None:
        from vllm.agent_tracker.tracker import get_session_tracker_registry
        reg = get_session_tracker_registry()
    return reg


@router.get("/v1/agent_tracker/sessions/{session_id}/opportunity")
async def get_opportunity(session_id: str, raw_request: Request):
    snap = _registry(raw_request).get_opportunity(session_id)
    if snap is None:
        return JSONResponse({"error": f"session {session_id!r} not found"},
                            status_code=404)
    return JSONResponse(snap)


@router.delete("/v1/agent_tracker/sessions/{session_id}")
async def delete_session(session_id: str, raw_request: Request):
    _registry(raw_request).delete(session_id)
    return JSONResponse({"deleted": session_id})


def attach_router(app: FastAPI) -> None:
    """Attach the agent_tracker router to a FastAPI app. Called from
    `vllm/entrypoints/openai/api_server.py::build_app`."""
    app.include_router(router)
