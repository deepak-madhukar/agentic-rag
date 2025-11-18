from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from app.core.dependecies import get_dependencies, Dependencies

router = APIRouter()

@router.post("/evaluate")
async def run_evaluation(deps: Dependencies = Depends(get_dependencies)):
    if not deps.is_ready():
        return JSONResponse(
            status_code=503,
            content={"detail": "System not ready"},
        )

    try:
        import asyncio

        from rag_pipeline.evaluator import RAGEvaluator
        from pathlib import Path

        evaluator = RAGEvaluator(deps.rag_pipeline, Path("results"))
        results = await evaluator.evaluate()

        return {
            "status": "completed",
            "results_file": "results/results.json",
            "chart_files": [
                "results/faithfulness_by_query.png",
                "results/latency_histogram.png",
            ],
            "summary": results.get("summary", {}),
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"detail": f"Evaluation failed: {str(e)}"},
        )