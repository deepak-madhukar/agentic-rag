from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from app.core.dependecies import get_dependencies, Dependencies
from app.service.trace_store import get_trace_store

router = APIRouter()

@router.get("/debug/trace")
async def get_latest_trace(deps: Dependencies = Depends(get_dependencies)):
    trace = get_trace_store().get_latest()

    if not trace:
        return JSONResponse(
            status_code=404,
            content={"detail": "No trace available yet"},
        )

    return trace