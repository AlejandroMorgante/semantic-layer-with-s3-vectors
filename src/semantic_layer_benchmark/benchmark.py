from __future__ import annotations

import csv
import json
import statistics
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from semantic_layer_benchmark.agent import (
    AgentRun,
    run_full_context,
    run_hybrid_context,
    run_knowledge_base_context,
    run_vector_context,
)
from semantic_layer_benchmark.aws_services import AwsServices

TARGET_STRATEGIES = (
    "full_markdown",
    "s3_vectors_kb",
    "neptune_graphrag_kb",
    "hybrid",
)
LEGACY_STRATEGIES = ("full_markdown", "direct_s3_vectors")
ALL_STRATEGIES = frozenset((*TARGET_STRATEGIES, *LEGACY_STRATEGIES))


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
    if strategy in {"s3_vectors", "direct_s3_vectors"}:
        cost += (
            run.retrieval_embedding_tokens * pricing.titan_embedding_input_per_million
        ) / 1_000_000
        cost += pricing.s3_vectors_query_per_million / 1_000_000
    return cost


def run_benchmark(
    services: AwsServices,
    catalog_markdown: str,
    questions: list[dict[str, Any]],
    model_id: str,
    top_k: int,
    repetitions: int,
    output_dir: Path,
    bucket: str | None = None,
    index: str | None = None,
    embedding_model_id: str | None = None,
    s3_vectors_kb_id: str | None = None,
    neptune_kb_id: str | None = None,
    strategies: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    selected_strategies = strategies or _default_strategies(
        bucket=bucket,
        index=index,
        embedding_model_id=embedding_model_id,
        s3_vectors_kb_id=s3_vectors_kb_id,
        neptune_kb_id=neptune_kb_id,
    )
    _validate_configuration(
        strategies=selected_strategies,
        bucket=bucket,
        index=index,
        embedding_model_id=embedding_model_id,
        s3_vectors_kb_id=s3_vectors_kb_id,
        neptune_kb_id=neptune_kb_id,
    )
    _validate_questions(questions)

    pricing = Pricing()
    results: list[dict[str, Any]] = []
    for repetition in range(1, repetitions + 1):
        for question in questions:
            for strategy in selected_strategies:
                run = _run_strategy(
                    strategy=strategy,
                    services=services,
                    model_id=model_id,
                    embedding_model_id=embedding_model_id,
                    bucket=bucket,
                    index=index,
                    s3_vectors_kb_id=s3_vectors_kb_id,
                    neptune_kb_id=neptune_kb_id,
                    question=question["question"],
                    catalog_markdown=catalog_markdown,
                    top_k=top_k,
                )
                expected = set(question["expected_tables"])
                selected = set(run.selected_tables)
                retrieved = set(run.retrieved_tables)
                metrics = _selection_metrics(expected, selected)
                result = {
                    "run_at": datetime.now(UTC).isoformat(),
                    "repetition": repetition,
                    "question_id": question["id"],
                    "category": question["category"],
                    "question": question["question"],
                    "strategy": strategy,
                    "expected_tables": sorted(expected),
                    "selected_tables": run.selected_tables,
                    "retrieved_tables": run.retrieved_tables,
                    "retrieval_scores": run.retrieval_scores,
                    "filters": run.filters,
                    "tools_used": run.tools_used,
                    "source_calls": len(run.tools_used),
                    "selection_correct": metrics["exact_match"],
                    "selection_exact_match": metrics["exact_match"],
                    "selection_precision": metrics["precision"],
                    "selection_recall": metrics["recall"],
                    "selection_f1": metrics["f1"],
                    "retrieval_recall": (
                        len(expected & retrieved) / len(expected) if retrieved else None
                    ),
                    "input_tokens": run.input_tokens,
                    "output_tokens": run.output_tokens,
                    "retrieval_embedding_tokens": run.retrieval_embedding_tokens,
                    "retrieval_latency_ms": round(run.retrieval_latency_ms, 2),
                    "total_latency_ms": round(run.total_latency_ms, 2),
                    "estimated_cost_usd": round(estimated_cost(run, strategy, pricing), 10),
                    "cost_estimate_scope": (
                        "model_and_direct_retrieval"
                        if strategy == "direct_s3_vectors"
                        else "model_only"
                    ),
                    "raw_answer": run.raw_answer,
                    "model_id": model_id,
                    "embedding_model_id": embedding_model_id,
                    "s3_vectors_kb_id": s3_vectors_kb_id,
                    "neptune_kb_id": neptune_kb_id,
                    "region": services.region,
                    "top_k": top_k,
                    "pricing_effective_date": pricing.effective_date,
                }
                results.append(result)
                print(
                    f"{question['id']}: {strategy} "
                    f"exact={result['selection_exact_match']} f1={result['selection_f1']:.3f} "
                    f"input_tokens={run.input_tokens}"
                )
    _write_results(results, output_dir, selected_strategies)
    return results


def _run_strategy(
    *,
    strategy: str,
    services: AwsServices,
    model_id: str,
    embedding_model_id: str | None,
    bucket: str | None,
    index: str | None,
    s3_vectors_kb_id: str | None,
    neptune_kb_id: str | None,
    question: str,
    catalog_markdown: str,
    top_k: int,
) -> AgentRun:
    if strategy == "full_markdown":
        return run_full_context(services, model_id, question, catalog_markdown)
    if strategy == "direct_s3_vectors":
        return run_vector_context(
            services,
            model_id,
            _required(embedding_model_id, "embedding_model_id"),
            _required(bucket, "bucket"),
            _required(index, "index"),
            question,
            top_k,
        )
    if strategy == "s3_vectors_kb":
        return run_knowledge_base_context(
            services,
            model_id,
            _required(s3_vectors_kb_id, "s3_vectors_kb_id"),
            question,
            top_k,
            source_name="s3_vectors_kb",
        )
    if strategy == "neptune_graphrag_kb":
        return run_knowledge_base_context(
            services,
            model_id,
            _required(neptune_kb_id, "neptune_kb_id"),
            question,
            top_k,
            source_name="neptune_graphrag_kb",
        )
    if strategy == "hybrid":
        return run_hybrid_context(
            services,
            model_id,
            _required(s3_vectors_kb_id, "s3_vectors_kb_id"),
            _required(neptune_kb_id, "neptune_kb_id"),
            question,
            catalog_markdown,
            top_k,
        )
    raise ValueError(f"Unsupported strategy: {strategy}")


def _default_strategies(
    *,
    bucket: str | None,
    index: str | None,
    embedding_model_id: str | None,
    s3_vectors_kb_id: str | None,
    neptune_kb_id: str | None,
) -> tuple[str, ...]:
    if s3_vectors_kb_id and neptune_kb_id:
        return TARGET_STRATEGIES
    if bucket and index and embedding_model_id:
        return LEGACY_STRATEGIES
    return ("full_markdown",)


def _validate_configuration(
    *,
    strategies: tuple[str, ...],
    bucket: str | None,
    index: str | None,
    embedding_model_id: str | None,
    s3_vectors_kb_id: str | None,
    neptune_kb_id: str | None,
) -> None:
    unknown = set(strategies) - ALL_STRATEGIES
    if unknown:
        raise ValueError(f"Unknown benchmark strategies: {', '.join(sorted(unknown))}")
    if len(strategies) != len(set(strategies)):
        raise ValueError("Benchmark strategies must not contain duplicates")
    if "direct_s3_vectors" in strategies and not (bucket and index and embedding_model_id):
        raise ValueError("direct_s3_vectors requires --bucket, --index, and --embedding-model-id")
    if {"s3_vectors_kb", "hybrid"} & set(strategies) and not s3_vectors_kb_id:
        raise ValueError("s3_vectors_kb and hybrid require --s3-vectors-kb-id")
    if {"neptune_graphrag_kb", "hybrid"} & set(strategies) and not neptune_kb_id:
        raise ValueError("neptune_graphrag_kb and hybrid require --neptune-kb-id")


def _validate_questions(questions: list[dict[str, Any]]) -> None:
    if not questions:
        raise ValueError("At least one benchmark question is required")
    required = {"id", "category", "question", "expected_tables"}
    for position, question in enumerate(questions, start=1):
        missing = required - set(question)
        if missing:
            raise ValueError(
                f"Question {position} is missing required fields: {', '.join(sorted(missing))}"
            )
        if not question["expected_tables"]:
            raise ValueError(f"Question {question['id']} must define expected_tables")


def _selection_metrics(expected: set[str], selected: set[str]) -> dict[str, float | bool]:
    true_positives = len(expected & selected)
    precision = true_positives / len(selected) if selected else 0.0
    recall = true_positives / len(expected)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "exact_match": selected == expected,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _required(value: str | None, name: str) -> str:
    if not value:
        raise ValueError(f"Missing required benchmark configuration: {name}")
    return value


def _write_results(
    results: list[dict[str, Any]], output_dir: Path, strategies: tuple[str, ...]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "raw-results.json").write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )
    fields = [
        "run_at",
        "repetition",
        "question_id",
        "category",
        "strategy",
        "selection_correct",
        "selection_exact_match",
        "selection_precision",
        "selection_recall",
        "selection_f1",
        "retrieval_recall",
        "source_calls",
        "input_tokens",
        "output_tokens",
        "retrieval_embedding_tokens",
        "retrieval_latency_ms",
        "total_latency_ms",
        "estimated_cost_usd",
        "cost_estimate_scope",
        "model_id",
        "embedding_model_id",
        "region",
        "top_k",
    ]
    with (output_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: result[field] for field in fields} for result in results)

    summary = [
        "# Benchmark summary",
        "",
        "Knowledge Base retrieval charges are not included because the Retrieve API does not ",
        "report an itemized per-request charge. Those rows contain model cost only.",
        "",
        "## Overall",
        "",
        "| Strategy | Exact match | Category-macro exact | Mean F1 | Mean input tokens | "
        "Mean latency | Cost scope |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for strategy in strategies:
        subset = [result for result in results if result["strategy"] == strategy]
        summary.append(_summary_row(strategy, subset))

    categories = list(dict.fromkeys(result["category"] for result in results))
    summary.extend(
        [
            "",
            "## Exact match by question category",
            "",
            "| Category | " + " | ".join(strategies) + " |",
            "| --- | " + " | ".join("---:" for _ in strategies) + " |",
        ]
    )
    for category in categories:
        cells = []
        for strategy in strategies:
            subset = [
                result
                for result in results
                if result["strategy"] == strategy and result["category"] == category
            ]
            accuracy = statistics.mean(bool(item["selection_exact_match"]) for item in subset)
            cells.append(f"{accuracy:.1%}")
        summary.append(f"| {category} | " + " | ".join(cells) + " |")

    hybrid = [result for result in results if result["strategy"] == "hybrid"]
    if hybrid:
        mean_source_calls = statistics.mean(item["source_calls"] for item in hybrid)
        summary.extend(
            [
                "",
                "## Hybrid routing",
                "",
                f"- Mean source calls: {mean_source_calls:.2f}",
                "- Routes used:",
                "",
            ]
        )
        route_counts: dict[str, int] = {}
        for result in hybrid:
            route = " -> ".join(result["tools_used"])
            route_counts[route] = route_counts.get(route, 0) + 1
        for route, count in sorted(route_counts.items()):
            summary.append(f"  - `{route}`: {count}")
        summary.extend(
            [
                "",
                "| Category | Most common route | Mean calls | Exact match |",
                "| --- | --- | ---: | ---: |",
            ]
        )
        for category in categories:
            category_results = [item for item in hybrid if item["category"] == category]
            category_routes: dict[str, int] = {}
            for item in category_results:
                route = " -> ".join(item["tools_used"])
                category_routes[route] = category_routes.get(route, 0) + 1
            common_route = max(category_routes, key=category_routes.get)
            mean_calls = statistics.mean(item["source_calls"] for item in category_results)
            accuracy = statistics.mean(
                bool(item["selection_exact_match"]) for item in category_results
            )
            summary.append(f"| {category} | `{common_route}` | {mean_calls:.2f} | {accuracy:.1%} |")

    (output_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    (output_dir / "run-metadata.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now(UTC).isoformat(),
                "records": len(results),
                "strategies": list(strategies),
                "pricing": asdict(Pricing()),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _summary_row(strategy: str, subset: list[dict[str, Any]]) -> str:
    accuracy = statistics.mean(bool(item["selection_exact_match"]) for item in subset)
    categories = {item["category"] for item in subset}
    macro_accuracy = statistics.mean(
        statistics.mean(
            bool(item["selection_exact_match"]) for item in subset if item["category"] == category
        )
        for category in categories
    )
    mean_f1 = statistics.mean(item["selection_f1"] for item in subset)
    mean_input_tokens = statistics.mean(item["input_tokens"] for item in subset)
    mean_latency_ms = statistics.mean(item["total_latency_ms"] for item in subset)
    scopes = sorted({item["cost_estimate_scope"] for item in subset})
    return (
        f"| {strategy} | {accuracy:.1%} | {macro_accuracy:.1%} | {mean_f1:.3f} | "
        f"{mean_input_tokens:.1f} | {mean_latency_ms:.1f} ms | {', '.join(scopes)} |"
    )
