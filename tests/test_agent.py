import pytest

from semantic_layer_benchmark.agent import _parse_table_selection, _validate_tool_input


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
