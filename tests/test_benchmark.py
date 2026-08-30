import pytest

from semantic_layer_benchmark.agent import AgentRun
from semantic_layer_benchmark.benchmark import Pricing, estimated_cost


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
    assert estimated_cost(run, "s3_vectors", pricing) == pytest.approx(expected)
