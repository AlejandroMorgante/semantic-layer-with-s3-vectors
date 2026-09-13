import pytest

from semantic_layer_benchmark.agent import (
    _extract_table_keys,
    _parse_table_selection,
    _validate_tool_input,
    run_hybrid_context,
    run_knowledge_base_context,
)
from semantic_layer_benchmark.aws_services import KnowledgeBaseSearchResult


class FakeBedrock:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = iter(responses)
        self.calls: list[dict] = []

    def converse(self, **kwargs: object) -> dict:
        self.calls.append(kwargs)
        return next(self.responses)


class FakeServices:
    def __init__(self, responses: list[dict]) -> None:
        self.bedrock = FakeBedrock(responses)
        self.retrievals: list[tuple[str, str, int]] = []

    def retrieve_knowledge_base(
        self, knowledge_base_id: str, query: str, top_k: int
    ) -> KnowledgeBaseSearchResult:
        self.retrievals.append((knowledge_base_id, query, top_k))
        return KnowledgeBaseSearchResult(
            documents=[
                {
                    "text": (
                        "Table: mexico_gold_logistics_delivery\n"
                        "Relationships:\n- DERIVED_FROM mexico_silver_orders_enriched"
                    ),
                    "metadata": {},
                    "score": 0.91,
                }
            ],
            latency_ms=12.5,
        )


def test_validate_tool_input_accepts_allowlisted_filters() -> None:
    result = _validate_tool_input(
        {
            "query": "net sales last quarter",
            "country": "argentina",
            "layer": "gold",
        }
    )
    assert result["country"] == "argentina"
    assert result["layer"] == "gold"


def test_validate_tool_input_rejects_unknown_country() -> None:
    with pytest.raises(ValueError, match="Unsupported country"):
        _validate_tool_input({"query": "sales", "country": "unknown", "layer": "gold"})


def test_parse_table_selection_handles_json_wrapping() -> None:
    assert _parse_table_selection('Result: {"tables": ["argentina_gold_sales_daily"]}') == [
        "argentina_gold_sales_daily"
    ]


def test_extract_table_keys_includes_relationship_targets_and_metadata() -> None:
    documents = [
        {
            "text": (
                "Table: mexico_gold_logistics_delivery\n"
                "Relationships:\n- DERIVED_FROM mexico_silver_orders_enriched"
            ),
            "metadata": {"table_key": "mexico_gold_logistics_delivery"},
        }
    ]
    assert _extract_table_keys(documents) == [
        "mexico_gold_logistics_delivery",
        "mexico_silver_orders_enriched",
    ]


def test_knowledge_base_agent_retrieves_then_uses_same_answer_contract() -> None:
    services = FakeServices(
        [
            {
                "output": {
                    "message": {
                        "content": [{"text": '{"tables": ["mexico_gold_logistics_delivery"]}'}]
                    }
                },
                "usage": {"inputTokens": 100, "outputTokens": 10},
            }
        ]
    )
    run = run_knowledge_base_context(
        services, "model", "kb-s3", "delivery delays", 5, "s3_vectors_kb"
    )
    assert services.retrievals == [("kb-s3", "delivery delays", 5)]
    assert run.selected_tables == ["mexico_gold_logistics_delivery"]
    assert run.tools_used == ["s3_vectors_kb"]
    assert run.retrieval_scores == [0.91]


def test_hybrid_agent_records_the_route_and_all_model_tokens() -> None:
    services = FakeServices(
        [
            {
                "stopReason": "tool_use",
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "toolUse": {
                                    "toolUseId": "tool-1",
                                    "name": "search_neptune_graphrag",
                                    "input": {"query": "Mexico delivery lineage"},
                                }
                            }
                        ],
                    }
                },
                "usage": {"inputTokens": 20, "outputTokens": 5},
            },
            {
                "stopReason": "end_turn",
                "output": {
                    "message": {
                        "content": [
                            {
                                "text": (
                                    '{"tables": ["mexico_silver_orders_enriched", '
                                    '"mexico_gold_logistics_delivery"]}'
                                )
                            }
                        ]
                    }
                },
                "usage": {"inputTokens": 40, "outputTokens": 8},
            },
        ]
    )
    run = run_hybrid_context(
        services,
        "model",
        "kb-s3",
        "kb-graph",
        "delivery lineage",
        "catalog",
        5,
    )
    assert services.retrievals == [("kb-graph", "Mexico delivery lineage", 5)]
    assert run.tools_used == ["search_neptune_graphrag"]
    assert run.input_tokens == 60
    assert run.output_tokens == 13
    assert set(run.retrieved_tables) == {
        "mexico_gold_logistics_delivery",
        "mexico_silver_orders_enriched",
    }


def test_hybrid_agent_gets_a_tool_free_final_turn_after_three_tool_rounds() -> None:
    tool_responses = []
    for index, name in enumerate(
        ["search_s3_vectors_kb", "search_neptune_graphrag", "load_full_markdown"],
        start=1,
    ):
        tool_responses.append(
            {
                "stopReason": "tool_use",
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "toolUse": {
                                    "toolUseId": f"tool-{index}",
                                    "name": name,
                                    "input": (
                                        {"query": "inventory"}
                                        if name != "load_full_markdown"
                                        else {}
                                    ),
                                }
                            }
                        ],
                    }
                },
                "usage": {"inputTokens": 10, "outputTokens": 2},
            }
        )
    services = FakeServices(
        tool_responses
        + [
            {
                "stopReason": "end_turn",
                "output": {
                    "message": {
                        "content": [{"text": '{"tables": ["mexico_gold_logistics_delivery"]}'}]
                    }
                },
                "usage": {"inputTokens": 50, "outputTokens": 5},
            }
        ]
    )

    run = run_hybrid_context(
        services,
        "model",
        "kb-s3",
        "kb-graph",
        "inventory",
        "Table: mexico_gold_logistics_delivery",
        5,
    )

    assert run.selected_tables == ["mexico_gold_logistics_delivery"]
    assert run.input_tokens == 80
    assert run.output_tokens == 11
    assert run.tools_used == [
        "search_s3_vectors_kb",
        "search_neptune_graphrag",
        "load_full_markdown",
    ]
    assert "toolConfig" not in services.bedrock.calls[-1]
