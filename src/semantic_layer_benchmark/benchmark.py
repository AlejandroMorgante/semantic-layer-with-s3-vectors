from __future__ import annotations

import csv
import json
import statistics
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from semantic_layer_benchmark.agent import AgentRun, run_full_context, run_vector_context
from semantic_layer_benchmark.aws_services import AwsServices


@dataclass(frozen=True)
class Pricing:
    effective_date: str = "2026-08-30"
    nova_micro_input_per_million: float = 0.035
    nova_micro_output_per_million: float = 0.14
    titan_embedding_input_per_million: float = 0.02
    s3_vectors_query_per_million: float = 2.50


def estimated_cost(run: AgentRun, strategy: str, pricing: Pricing) -> float:
    cost = (
        run.input_tokens * pricing.nova_micro_input_per_million
        + run.output_tokens * pricing.nova_micro_output_per_million
    ) / 1_000_000
    if strategy == "s3_vectors":
        cost += (
            run.retrieval_embedding_tokens * pricing.titan_embedding_input_per_million
        ) / 1_000_000
        cost += pricing.s3_vectors_query_per_million / 1_000_000
    return cost


def run_benchmark(
    services: AwsServices,
    bucket: str,
    index: str,
    catalog_markdown: str,
    questions: list[dict[str, Any]],
    model_id: str,
    embedding_model_id: str,
    top_k: int,
    repetitions: int,
    output_dir: Path,
) -> list[dict[str, Any]]:
    pricing = Pricing()
    results: list[dict[str, Any]] = []
    for repetition in range(1, repetitions + 1):
        for question in questions:
            for strategy in ("full_catalog", "s3_vectors"):
                if strategy == "full_catalog":
                    run = run_full_context(
                        services,
                        model_id,
                        question["question"],
                        catalog_markdown,
                    )
                else:
                    run = run_vector_context(
                        services,
                        model_id,
                        embedding_model_id,
                        bucket,
                        index,
                        question["question"],
                        top_k,
                    )
                expected = set(question["expected_tables"])
                selected = set(run.selected_tables)
                retrieved = set(run.retrieved_tables)
                result = {
                    "run_at": datetime.now(UTC).isoformat(),
                    "repetition": repetition,
                    "question_id": question["id"],
                    "question": question["question"],
                    "strategy": strategy,
                    "expected_tables": sorted(expected),
                    "selected_tables": run.selected_tables,
                    "retrieved_tables": run.retrieved_tables,
                    "filters": run.filters,
                    "selection_correct": expected.issubset(selected),
                    "retrieval_recall": (
                        len(expected & retrieved) / len(expected)
                        if strategy == "s3_vectors"
                        else None
                    ),
                    "input_tokens": run.input_tokens,
                    "output_tokens": run.output_tokens,
                    "retrieval_embedding_tokens": run.retrieval_embedding_tokens,
                    "retrieval_latency_ms": round(run.retrieval_latency_ms, 2),
                    "total_latency_ms": round(run.total_latency_ms, 2),
                    "estimated_cost_usd": round(estimated_cost(run, strategy, pricing), 10),
                    "raw_answer": run.raw_answer,
                    "model_id": model_id,
                    "embedding_model_id": embedding_model_id,
                    "region": services.region,
                    "top_k": top_k,
                    "pricing_effective_date": pricing.effective_date,
                }
                results.append(result)
                print(
                    f"{question['id']}: {strategy} "
                    f"correct={result['selection_correct']} input_tokens={run.input_tokens}"
                )
    _write_results(results, output_dir)
    return results


def _write_results(results: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "raw-results.json").write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )
    fields = [
        "run_at",
        "repetition",
        "question_id",
        "strategy",
        "selection_correct",
        "retrieval_recall",
        "input_tokens",
        "output_tokens",
        "retrieval_embedding_tokens",
        "retrieval_latency_ms",
        "total_latency_ms",
        "estimated_cost_usd",
        "model_id",
        "embedding_model_id",
        "region",
        "top_k",
    ]
    with (output_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: result[field] for field in fields} for result in results)

    summary = ["# Benchmark summary", ""]
    for strategy in ("full_catalog", "s3_vectors"):
        subset = [result for result in results if result["strategy"] == strategy]
        accuracy = sum(bool(item["selection_correct"]) for item in subset) / len(subset)
        mean_input_tokens = statistics.mean(item["input_tokens"] for item in subset)
        mean_latency_ms = statistics.mean(item["total_latency_ms"] for item in subset)
        mean_cost = statistics.mean(item["estimated_cost_usd"] for item in subset)
        summary.extend(
            [
                f"## {strategy}",
                "",
                f"- Selection accuracy: {accuracy:.1%}",
                f"- Mean input tokens: {mean_input_tokens:.1f}",
                f"- Mean total latency: {mean_latency_ms:.1f} ms",
                f"- Mean estimated cost: ${mean_cost:.8f}",
                "",
            ]
        )
    (output_dir / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    (output_dir / "run-metadata.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now(UTC).isoformat(),
                "records": len(results),
                "pricing": asdict(Pricing()),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
