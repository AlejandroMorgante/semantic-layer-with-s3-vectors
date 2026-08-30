from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any

from semantic_layer_benchmark.aws_services import AwsServices
from semantic_layer_benchmark.catalog import COUNTRIES, LAYERS

SYSTEM_PROMPT = """You select the smallest set of data-platform tables needed to answer a
business question. Prefer certified gold tables for business metrics. Use silver only when the
question requests cleaned pre-aggregation records, and bronze only when it explicitly requests
raw source data. Return only JSON in this exact form: {"tables": ["table_key"]}."""

VECTOR_SYSTEM_PROMPT = (
    SYSTEM_PROMPT
    + """
You do not have the table catalog in context. You must call search_tables exactly once before
selecting tables. Extract the country and medallion layer conservatively. Keep the semantic query
faithful to the user's question."""
)

FINAL_VECTOR_SYSTEM_PROMPT = (
    SYSTEM_PROMPT
    + """
The search_tables tool has already returned the only context available for this question. Do not
call a tool again. Select the best table or tables from that result and return the required JSON."""
)


@dataclass(frozen=True)
class AgentRun:
    selected_tables: list[str]
    retrieved_tables: list[str]
    filters: dict[str, str]
    input_tokens: int
    output_tokens: int
    retrieval_embedding_tokens: int
    retrieval_latency_ms: float
    total_latency_ms: float
    raw_answer: str


def run_full_context(
    services: AwsServices,
    model_id: str,
    question: str,
    catalog_markdown: str,
    max_tokens: int = 256,
) -> AgentRun:
    started = time.perf_counter()
    response = services.bedrock.converse(
        modelId=model_id,
        system=[{"text": SYSTEM_PROMPT}],
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "text": (
                            f"Business question:\n{question}\n\n"
                            f"Complete table catalog:\n{catalog_markdown}"
                        )
                    }
                ],
            }
        ],
        inferenceConfig={"maxTokens": max_tokens, "temperature": 0},
    )
    answer = _response_text(response)
    usage = response.get("usage", {})
    return AgentRun(
        selected_tables=_parse_table_selection(answer),
        retrieved_tables=[],
        filters={},
        input_tokens=int(usage.get("inputTokens", 0)),
        output_tokens=int(usage.get("outputTokens", 0)),
        retrieval_embedding_tokens=0,
        retrieval_latency_ms=0,
        total_latency_ms=(time.perf_counter() - started) * 1000,
        raw_answer=answer,
    )


def run_vector_context(
    services: AwsServices,
    model_id: str,
    embedding_model_id: str,
    bucket: str,
    index: str,
    question: str,
    top_k: int,
    max_tokens: int = 256,
) -> AgentRun:
    started = time.perf_counter()
    tool_config = _tool_config()
    messages: list[dict[str, Any]] = [{"role": "user", "content": [{"text": question}]}]
    first = services.bedrock.converse(
        modelId=model_id,
        system=[{"text": VECTOR_SYSTEM_PROMPT}],
        messages=messages,
        toolConfig=tool_config,
        inferenceConfig={"maxTokens": max_tokens, "temperature": 0},
    )
    if first.get("stopReason") != "tool_use":
        raise ValueError(f"Model did not call search_tables: {first.get('stopReason')}")

    tool_use = next(
        block["toolUse"] for block in first["output"]["message"]["content"] if "toolUse" in block
    )
    tool_input = _validate_tool_input(tool_use["input"])
    filters = {key: tool_input[key] for key in ("country", "layer")}
    search = services.search_tables(
        bucket=bucket,
        index=index,
        query=tool_input["query"],
        embedding_model_id=embedding_model_id,
        filters=filters,
        top_k=top_k,
    )
    print(f"search_tables filters={filters} retrieved={[table['key'] for table in search.tables]}")
    messages.extend(
        [
            first["output"]["message"],
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": tool_use["toolUseId"],
                            "content": [{"json": {"tables": search.tables}}],
                        }
                    }
                ],
            },
        ]
    )
    final = services.bedrock.converse(
        modelId=model_id,
        system=[{"text": FINAL_VECTOR_SYSTEM_PROMPT}],
        messages=messages,
        toolConfig=tool_config,
        inferenceConfig={"maxTokens": max_tokens, "temperature": 0},
    )
    if final.get("stopReason") == "tool_use":
        raise ValueError("Model attempted a second search_tables call after receiving results")
    answer = _response_text(final)
    first_usage = first.get("usage", {})
    final_usage = final.get("usage", {})
    return AgentRun(
        selected_tables=_parse_table_selection(answer),
        retrieved_tables=[table["key"] for table in search.tables],
        filters=filters,
        input_tokens=int(first_usage.get("inputTokens", 0))
        + int(final_usage.get("inputTokens", 0)),
        output_tokens=int(first_usage.get("outputTokens", 0))
        + int(final_usage.get("outputTokens", 0)),
        retrieval_embedding_tokens=search.embedding_input_tokens,
        retrieval_latency_ms=search.latency_ms,
        total_latency_ms=(time.perf_counter() - started) * 1000,
        raw_answer=answer,
    )


def _tool_config() -> dict[str, Any]:
    return {
        "tools": [
            {
                "toolSpec": {
                    "name": "search_tables",
                    "description": (
                        "Find semantically relevant table descriptions, restricting the search "
                        "with country and medallion layer metadata."
                    ),
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "Semantic description of the requested data.",
                                },
                                "country": {
                                    "type": "string",
                                    "enum": sorted(COUNTRIES),
                                },
                                "layer": {"type": "string", "enum": list(LAYERS)},
                            },
                            "required": ["query", "country", "layer"],
                            "additionalProperties": False,
                        }
                    },
                }
            }
        ]
    }


def _validate_tool_input(payload: dict[str, Any]) -> dict[str, str]:
    query = payload.get("query")
    country = payload.get("country")
    layer = payload.get("layer")
    if not isinstance(query, str) or not query.strip() or len(query) > 1_000:
        raise ValueError("search_tables query must be a non-empty string under 1,000 characters")
    if country not in COUNTRIES:
        raise ValueError(f"Unsupported country: {country}")
    if layer not in LAYERS:
        raise ValueError(f"Unsupported layer: {layer}")
    return {"query": query.strip(), "country": country, "layer": layer}


def _response_text(response: dict[str, Any]) -> str:
    return "".join(
        block.get("text", "") for block in response["output"]["message"]["content"]
    ).strip()


def _parse_table_selection(answer: str) -> list[str]:
    try:
        payload = json.loads(answer)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", answer, flags=re.DOTALL)
        if not match:
            raise ValueError(f"Model did not return JSON: {answer}") from None
        payload = json.loads(match.group(0))
    tables = payload.get("tables")
    if not isinstance(tables, list) or not all(isinstance(item, str) for item in tables):
        raise ValueError(f"Invalid table selection: {payload}")
    return tables
