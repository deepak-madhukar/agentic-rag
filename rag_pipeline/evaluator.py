import asyncio
import json
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


class RAGEvaluator:
    def __init__(self, pipeline, results_dir: Path):
        self.pipeline = pipeline
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    async def evaluate(self) -> dict:
        queries = self._get_benchmark_queries()
        results = []

        for idx, query in enumerate(queries):
            logger.info(f"Evaluating query {idx + 1}/{len(queries)}: {query[:50]}...")

            synthesis_result, trace = self.pipeline.process_query(
                request_id=f"eval_{idx}",
                query=query,
                user_role="Admin",
            )

            result_item = {
                "query_id": idx,
                "query": query,
                "answer": synthesis_result.answer,
                "faithfulness": 1.0 - synthesis_result.hallucination_score,
                "relevance": min(len(synthesis_result.citations) / 3, 1.0),
                "hallucination_score": synthesis_result.hallucination_score,
                "latency_ms": trace.total_time_ms,
                "citations_count": len(synthesis_result.citations),
            }
            results.append(result_item)

        summary = self._compute_summary(results)

        output = {
            "benchmark_results": results,
            "summary": summary,
        }

        results_file = self.results_dir / "results.json"
        with open(results_file, "w") as f:
            json.dump(output, f, indent=2)

        self._generate_charts(results)

        logger.info(f"Evaluation complete. Results saved to {results_file}")
        return output

    def _get_benchmark_queries(self) -> list[str]:
        return [
            "What are the main features of ProductA?",
            "How do we deploy to production?",
            "What tickets are assigned to the engineering team?",
            "Explain the architecture of the system",
            "What were the recent changes in Q4 2024?",
            "What is the product roadmap?",
            "How do we handle security?",
            "What are the team responsibilities?",
            "Describe the deployment process",
            "What are the key metrics?",
            "How do we manage issues?",
            "What is the API specification?",
            "How do we monitor systems?",
            "What is the disaster recovery plan?",
            "What training materials are available?",
        ]

    def _compute_summary(self, results: list[dict]) -> dict:
        df = pd.DataFrame(results)

        return {
            "total_queries": len(results),
            "avg_faithfulness": float(df["faithfulness"].mean()),
            "avg_relevance": float(df["relevance"].mean()),
            "avg_hallucination": float(df["hallucination_score"].mean()),
            "avg_latency_ms": float(df["latency_ms"].mean()),
            "median_latency_ms": float(df["latency_ms"].median()),
            "p95_latency_ms": float(df["latency_ms"].quantile(0.95)),
        }

    def _generate_charts(self, results: list[dict]) -> None:
        df = pd.DataFrame(results)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.bar(df["query_id"], df["faithfulness"], color="steelblue", alpha=0.7)
        ax1.set_xlabel("Query ID")
        ax1.set_ylabel("Faithfulness Score")
        ax1.set_title("Faithfulness by Query")
        ax1.set_ylim([0, 1])
        ax1.grid(axis="y", alpha=0.3)

        ax2.hist(df["latency_ms"], bins=10, color="coral", alpha=0.7, edgecolor="black")
        ax2.set_xlabel("Latency (ms)")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Latency Distribution")
        ax2.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        chart_file = self.results_dir / "faithfulness_by_query.png"
        plt.savefig(chart_file, dpi=100, bbox_inches="tight")
        logger.info(f"Saved chart: {chart_file}")

        plt.close()

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(df["latency_ms"], bins=10, color="mediumseagreen", alpha=0.7, edgecolor="black")
        ax.set_xlabel("Latency (ms)")
        ax.set_ylabel("Frequency")
        ax.set_title("Latency Histogram")
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        hist_file = self.results_dir / "latency_histogram.png"
        plt.savefig(hist_file, dpi=100, bbox_inches="tight")
        logger.info(f"Saved chart: {hist_file}")

        plt.close()


async def run_evaluation():
    from pathlib import Path
    from ingestion_pipeline.index_builder import IndexBuilder

    index_dir = Path("indexes/store")
    results_dir = Path("rag_test")

    if not index_dir.exists():
        logger.error("Index not found. Run ingestion pipeline first.")
        return

    index_builder = IndexBuilder(index_dir)
    from rag_pipeline.pipeline import RAGPipeline

    pipeline = RAGPipeline(index_builder)
    evaluator = RAGEvaluator(pipeline, results_dir)

    await evaluator.evaluate()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_evaluation())
