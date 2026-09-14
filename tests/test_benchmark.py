import pytest

import semantic_layer_benchmark.benchmark as benchmark_module
from semantic_layer_benchmark.agent import AgentRun
from semantic_layer_benchmark.benchmark import (
    Pricing,
    _counterbalanced_order,
    _retrieval_recall,
    _selection_metrics,
    estimated_cost,
    run_benchmark,
)


def test_vector_cost_includes_embedding_and_query_request() -> None:
    run = AgentRun(
        selected_tables=[],
        retrieved_tables=[],
        filters={},
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        retrieval_embedding_tokens=1_000_000,
        retrieval_latency_ms=0,
        total_latency_ms=0,
        raw_answer="",
    )
    pricing = Pricing()
    expected = 0.035 + 0.14 + 0.02 + 0.0000025
    assert estimated_cost(
        run,
        "direct_s3_vectors",
        pricing,
        "us.amazon.nova-micro-v1:0",
        "amazon.titan-embed-text-v2:0",
    ) == pytest.approx(expected)


def test_cost_is_unavailable_for_unconfigured_model() -> None:
    run = AgentRun(
        selected_tables=[],
        retrieved_tables=[],
        filters={},
        input_tokens=100,
        output_tokens=10,
        retrieval_embedding_tokens=0,
        retrieval_latency_ms=0,
        total_latency_ms=0,
        raw_answer="",
    )

    assert estimated_cost(run, "full_markdown", Pricing(), "other-model") is None


def test_selection_metrics_penalize_extra_tables() -> None:
    metrics = _selection_metrics({"expected"}, {"expected", "extra"})
    assert metrics == {
        "exact_match": False,
        "precision": 0.5,
        "recall": 1.0,
        "f1": pytest.approx(2 / 3),
    }


def test_empty_retrieval_is_zero_for_retrieval_strategies_and_na_for_full_context() -> None:
    expected = {"required"}

    assert _retrieval_recall("s3_vectors_kb", expected, set()) == 0.0
    assert _retrieval_recall("neptune_graphrag_kb", expected, set()) == 0.0
    assert _retrieval_recall("hybrid", expected, set()) == 0.0
    assert _retrieval_recall("full_markdown", expected, set()) is None


def test_strategy_order_is_counterbalanced() -> None:
    strategies = ("a", "b", "c", "d")

    assert _counterbalanced_order(strategies, 0) == ("a", "b", "c", "d")
    assert _counterbalanced_order(strategies, 1) == ("b", "c", "d", "a")
    assert _counterbalanced_order(strategies, 4) == strategies


@pytest.mark.parametrize(("top_k", "repetitions"), [(0, 1), (1, 0), (-1, 1), (1, -1)])
def test_benchmark_rejects_invalid_paid_run_parameters_before_execution(
    tmp_path, top_k: int, repetitions: int
) -> None:
    with pytest.raises(ValueError, match="must be at least 1"):
        run_benchmark(
            services=None,  # type: ignore[arg-type]
            catalog_markdown="catalog",
            questions=[
                {
                    "id": "q1",
                    "category": "semantic_lookup",
                    "question": "Which table?",
                    "expected_tables": ["required"],
                }
            ],
            model_id="model",
            top_k=top_k,
            repetitions=repetitions,
            output_dir=tmp_path,
        )


def test_benchmark_treats_complete_coverage_with_extra_context_as_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    def fake_run_strategy(**kwargs: object) -> AgentRun:
        return AgentRun(
            selected_tables=["required", "extra"],
            retrieved_tables=[],
            filters={},
            input_tokens=10,
            output_tokens=2,
            retrieval_embedding_tokens=0,
            retrieval_latency_ms=0,
            total_latency_ms=1,
            raw_answer="",
            tools_used=[str(kwargs["strategy"])],
        )

    monkeypatch.setattr(benchmark_module, "_run_strategy", fake_run_strategy)
    services = type("FakeServices", (), {"region": "us-east-1"})()

    results = run_benchmark(
        services=services,  # type: ignore[arg-type]
        catalog_markdown="catalog",
        questions=[
            {
                "id": "q1",
                "category": "semantic_lookup",
                "question": "Which table?",
                "expected_tables": ["required"],
            }
        ],
        model_id="us.amazon.nova-micro-v1:0",
        top_k=5,
        repetitions=1,
        output_dir=tmp_path,
        s3_vectors_kb_id="kb-s3",
        strategies=("full_markdown", "s3_vectors_kb"),
    )

    assert [result["task_success"] for result in results] == [True, True]
    assert [result["selection_exact_match"] for result in results] == [False, False]
    assert [result["retrieval_recall"] for result in results] == [None, 0.0]
    assert [result["strategy_execution_order"] for result in results] == [1, 2]
    assert "Task success" in (tmp_path / "summary.md").read_text(encoding="utf-8")
