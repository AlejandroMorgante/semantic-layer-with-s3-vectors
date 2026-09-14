from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from semantic_layer_benchmark.aws_services import AwsServices
from semantic_layer_benchmark.catalog import COUNTRIES, LAYERS

HYBRID_ROUTING_POLICY = "prompt_guided_source_affordances_v1"

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

KNOWLEDGE_BASE_SYSTEM_PROMPT = (
    SYSTEM_PROMPT
    + """
Use only the Knowledge Base retrieval results included in the user message. They may contain
several chunks and explicit table relationships. Do not invent table names that are absent from
that context."""
)

HYBRID_SYSTEM_PROMPT = (
    SYSTEM_PROMPT
    + """
You can access three representations of the same catalog. You must call at least one tool before
answering. Use the smallest sufficient source set: semantic retrieval is useful for direct lookup,
graph retrieval is useful when explicit relationships or multi-hop lineage matter, and the full
Markdown catalog is useful when exhaustive global context is required. You may call another source
after inspecting the first result, but do not make redundant calls. Never assume one source is
always best."""
)

HYBRID_FINAL_SYSTEM_PROMPT = (
    SYSTEM_PROMPT
    + """
You have already consulted the allowed knowledge sources and their results are present in the
conversation. Do not request another tool. Synthesize the smallest sufficient table set and return
the required JSON now."""
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
    tools_used: list[str] = field(default_factory=list)
    retrieval_scores: list[float] = field(default_factory=list)


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
        tools_used=["full_markdown"],
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
        tools_used=["direct_s3_vectors"],
    )


def run_knowledge_base_context(
    services: AwsServices,
    model_id: str,
    knowledge_base_id: str,
    question: str,
    top_k: int,
    source_name: str,
    max_tokens: int = 256,
) -> AgentRun:
    started = time.perf_counter()
    search = services.retrieve_knowledge_base(
        knowledge_base_id=knowledge_base_id,
        query=question,
        top_k=top_k,
    )
    response = services.bedrock.converse(
        modelId=model_id,
        system=[{"text": KNOWLEDGE_BASE_SYSTEM_PROMPT}],
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "text": (
                            f"Business question:\n{question}\n\n"
                            "Knowledge Base retrieval results:\n"
                            f"{_format_documents(search.documents)}"
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
        retrieved_tables=_extract_table_keys(search.documents),
        filters={},
        input_tokens=int(usage.get("inputTokens", 0)),
        output_tokens=int(usage.get("outputTokens", 0)),
        retrieval_embedding_tokens=0,
        retrieval_latency_ms=search.latency_ms,
        total_latency_ms=(time.perf_counter() - started) * 1000,
        raw_answer=answer,
        tools_used=[source_name],
        retrieval_scores=[
            float(document["score"])
            for document in search.documents
            if document.get("score") is not None
        ],
    )


def run_hybrid_context(
    services: AwsServices,
    model_id: str,
    s3_vectors_kb_id: str,
    neptune_kb_id: str,
    question: str,
    catalog_markdown: str,
    top_k: int,
    max_tokens: int = 256,
    max_turns: int = 3,
) -> AgentRun:
    started = time.perf_counter()
    messages: list[dict[str, Any]] = [{"role": "user", "content": [{"text": question}]}]
    tool_config = _hybrid_tool_config()
    input_tokens = 0
    output_tokens = 0
    retrieval_latency_ms = 0.0
    retrieved_tables: list[str] = []
    retrieval_scores: list[float] = []
    tools_used: list[str] = []

    for _ in range(max_turns):
        response = services.bedrock.converse(
            modelId=model_id,
            system=[{"text": HYBRID_SYSTEM_PROMPT}],
            messages=messages,
            toolConfig=tool_config,
            inferenceConfig={"maxTokens": max_tokens, "temperature": 0},
        )
        usage = response.get("usage", {})
        input_tokens += int(usage.get("inputTokens", 0))
        output_tokens += int(usage.get("outputTokens", 0))
        if response.get("stopReason") != "tool_use":
            if not tools_used:
                raise ValueError("Hybrid agent answered without consulting a knowledge source")
            answer = _response_text(response)
            return AgentRun(
                selected_tables=_parse_table_selection(answer),
                retrieved_tables=list(dict.fromkeys(retrieved_tables)),
                filters={},
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                retrieval_embedding_tokens=0,
                retrieval_latency_ms=retrieval_latency_ms,
                total_latency_ms=(time.perf_counter() - started) * 1000,
                raw_answer=answer,
                tools_used=tools_used,
                retrieval_scores=retrieval_scores,
            )

        tool_results = []
        for block in response["output"]["message"]["content"]:
            if "toolUse" not in block:
                continue
            tool_use = block["toolUse"]
            name = tool_use["name"]
            tools_used.append(name)
            if name == "load_full_markdown":
                documents = [{"text": catalog_markdown, "metadata": {}, "score": None}]
            elif name in {"search_s3_vectors_kb", "search_neptune_graphrag"}:
                query = _validate_hybrid_query(tool_use.get("input", {}))
                knowledge_base_id = (
                    s3_vectors_kb_id if name == "search_s3_vectors_kb" else neptune_kb_id
                )
                search = services.retrieve_knowledge_base(
                    knowledge_base_id=knowledge_base_id,
                    query=query,
                    top_k=top_k,
                )
                documents = search.documents
                retrieval_latency_ms += search.latency_ms
                retrieval_scores.extend(
                    float(document["score"])
                    for document in documents
                    if document.get("score") is not None
                )
            else:
                raise ValueError(f"Hybrid agent requested an unsupported tool: {name}")
            retrieved_tables.extend(_extract_table_keys(documents))
            tool_results.append(
                {
                    "toolResult": {
                        "toolUseId": tool_use["toolUseId"],
                        "content": [{"json": {"documents": documents}}],
                    }
                }
            )
        if not tool_results:
            raise ValueError("Hybrid agent returned tool_use without a tool request")
        messages.extend(
            [
                response["output"]["message"],
                {"role": "user", "content": tool_results},
            ]
        )

    # A tool-using model can legitimately spend one round on each of the three
    # representations. Give it one final, tool-free turn so the bounded router
    # always has an opportunity to synthesize an answer after the last result.
    final = services.bedrock.converse(
        modelId=model_id,
        system=[{"text": HYBRID_FINAL_SYSTEM_PROMPT}],
        messages=messages,
        inferenceConfig={"maxTokens": max_tokens, "temperature": 0},
    )
    usage = final.get("usage", {})
    input_tokens += int(usage.get("inputTokens", 0))
    output_tokens += int(usage.get("outputTokens", 0))
    answer = _response_text(final)
    return AgentRun(
        selected_tables=_parse_table_selection(answer),
        retrieved_tables=list(dict.fromkeys(retrieved_tables)),
        filters={},
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        retrieval_embedding_tokens=0,
        retrieval_latency_ms=retrieval_latency_ms,
        total_latency_ms=(time.perf_counter() - started) * 1000,
        raw_answer=answer,
        tools_used=tools_used,
        retrieval_scores=retrieval_scores,
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


def _hybrid_tool_config() -> dict[str, Any]:
    query_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "A retrieval query faithful to the business question.",
            }
        },
        "required": ["query"],
        "additionalProperties": False,
    }
    return {
        "tools": [
            {
                "toolSpec": {
                    "name": "load_full_markdown",
                    "description": "Load the complete table catalog for exhaustive inspection.",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {},
                            "additionalProperties": False,
                        }
                    },
                }
            },
            {
                "toolSpec": {
                    "name": "search_s3_vectors_kb",
                    "description": (
                        "Semantically retrieve relevant catalog documents from the Bedrock "
                        "Knowledge Base backed by Amazon S3 Vectors."
                    ),
                    "inputSchema": {"json": query_schema},
                }
            },
            {
                "toolSpec": {
                    "name": "search_neptune_graphrag",
                    "description": (
                        "Retrieve catalog context from the Bedrock Knowledge Base backed by "
                        "Neptune Analytics GraphRAG, including relevant relationships."
                    ),
                    "inputSchema": {"json": query_schema},
                }
            },
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


def _validate_hybrid_query(payload: dict[str, Any]) -> str:
    query = payload.get("query")
    if not isinstance(query, str) or not query.strip() or len(query) > 1_000:
        raise ValueError("Knowledge Base query must be a non-empty string under 1,000 characters")
    return query.strip()


def _format_documents(documents: list[dict[str, Any]]) -> str:
    if not documents:
        return "No documents were retrieved."
    chunks = []
    for position, document in enumerate(documents, start=1):
        score = document.get("score")
        score_text = f" score={score:.6f}" if isinstance(score, int | float) else ""
        chunks.append(f"[Result {position}{score_text}]\n{document.get('text', '')}")
    return "\n\n".join(chunks)


def _extract_table_keys(documents: list[dict[str, Any]]) -> list[str]:
    keys: list[str] = []
    pattern = re.compile(
        r"\b(?:argentina|mexico|united_states)_(?:bronze|silver|gold)_[a-z0-9_]+\b",
        flags=re.IGNORECASE,
    )
    for document in documents:
        metadata = document.get("metadata", {})
        if isinstance(metadata, dict):
            for field_name in ("table_key", "key", "table"):
                value = metadata.get(field_name)
                if isinstance(value, str) and pattern.fullmatch(value):
                    keys.append(value.lower())
        text = document.get("text", "")
        if isinstance(text, str):
            keys.extend(match.group(0).lower() for match in pattern.finditer(text))
    return list(dict.fromkeys(keys))


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
