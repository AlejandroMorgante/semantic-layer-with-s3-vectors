from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from semantic_layer_benchmark.catalog import TableRecord

EMBEDDING_DIMENSION = 256


@dataclass(frozen=True)
class EmbeddingResult:
    vector: list[float]
    input_tokens: int


@dataclass(frozen=True)
class SearchResult:
    tables: list[dict[str, Any]]
    embedding_input_tokens: int
    latency_ms: float


@dataclass(frozen=True)
class KnowledgeBaseSearchResult:
    documents: list[dict[str, Any]]
    latency_ms: float


@dataclass(frozen=True)
class IngestionResult:
    ingestion_job_id: str
    status: str
    statistics: dict[str, int]


class AwsServices:
    def __init__(self, profile: str | None = None, region: str | None = None) -> None:
        selected_profile = profile or os.getenv("AWS_PROFILE") or None
        selected_region = region or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
        self.session = boto3.Session(
            profile_name=selected_profile,
            region_name=selected_region,
        )
        if not self.session.region_name:
            raise ValueError("Set AWS_REGION, AWS_DEFAULT_REGION, or configure a profile region")
        config = Config(
            retries={"total_max_attempts": 5, "mode": "adaptive"},
            connect_timeout=5,
            read_timeout=60,
            max_pool_connections=20,
            user_agent_appid="s3-vectors-semantic-layer-benchmark/0.1.0",
        )
        self.region = self.session.region_name
        self.sts = self.session.client("sts", config=config)
        self.bedrock = self.session.client("bedrock-runtime", config=config)
        self.bedrock_agent = self.session.client("bedrock-agent", config=config)
        self.bedrock_agent_runtime = self.session.client("bedrock-agent-runtime", config=config)
        self.neptune_graph = self.session.client("neptune-graph", config=config)
        self.s3 = self.session.client("s3", config=config)
        self.s3vectors = self.session.client("s3vectors", config=config)

    def identity(self) -> dict[str, Any]:
        return self.sts.get_caller_identity()

    def embed(self, text: str, model_id: str) -> EmbeddingResult:
        response = self.bedrock.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(
                {
                    "inputText": text,
                    "dimensions": EMBEDDING_DIMENSION,
                    "normalize": True,
                }
            ),
        )
        payload = json.loads(response["body"].read())
        return EmbeddingResult(
            vector=payload["embedding"],
            input_tokens=int(payload.get("inputTextTokenCount", 0)),
        )

    def index_catalog(
        self,
        bucket: str,
        index: str,
        records: list[TableRecord],
        embedding_model_id: str,
        workers: int = 4,
    ) -> dict[str, int]:
        record_keys = [record.key for record in records]
        if len(record_keys) != len(set(record_keys)):
            raise ValueError("Catalog records must have unique keys")

        existing_keys = set(self._list_vector_keys(bucket, index))

        def vectorize(record: TableRecord) -> tuple[dict[str, Any], int]:
            embedded = self.embed(record.context(), embedding_model_id)
            return (
                {
                    "key": record.key,
                    "data": {"float32": embedded.vector},
                    "metadata": {
                        "country": record.country,
                        "layer": record.layer,
                        "domain": record.domain,
                        "content": record.context(),
                    },
                },
                embedded.input_tokens,
            )

        with ThreadPoolExecutor(max_workers=workers) as executor:
            embedded_records = list(executor.map(vectorize, records))
        vectors = [item[0] for item in embedded_records]
        for start in range(0, len(vectors), 100):
            self.s3vectors.put_vectors(
                vectorBucketName=bucket,
                indexName=index,
                vectors=vectors[start : start + 100],
            )
        stale_keys = sorted(existing_keys - set(record_keys))
        self._delete_vector_keys(bucket, index, stale_keys)
        return {
            "vectors_indexed": len(vectors),
            "vectors_deleted": len(stale_keys),
            "embedding_input_tokens": sum(item[1] for item in embedded_records),
            "put_requests": (len(vectors) + 99) // 100,
        }

    def search_tables(
        self,
        bucket: str,
        index: str,
        query: str,
        embedding_model_id: str,
        filters: dict[str, str],
        top_k: int,
    ) -> SearchResult:
        started = time.perf_counter()
        embedded = self.embed(query, embedding_model_id)
        conditions = [{key: {"$eq": value}} for key, value in sorted(filters.items())]
        query_filter: dict[str, Any] | None = None
        if len(conditions) == 1:
            query_filter = conditions[0]
        elif conditions:
            query_filter = {"$and": conditions}
        request: dict[str, Any] = {
            "vectorBucketName": bucket,
            "indexName": index,
            "queryVector": {"float32": embedded.vector},
            "topK": top_k,
            "returnDistance": True,
            "returnMetadata": True,
        }
        if query_filter is not None:
            request["filter"] = query_filter
        response = self.s3vectors.query_vectors(**request)
        tables = [
            {
                "key": vector["key"],
                "distance": vector.get("distance"),
                **vector.get("metadata", {}),
            }
            for vector in response.get("vectors", [])
        ]
        return SearchResult(
            tables=tables,
            embedding_input_tokens=embedded.input_tokens,
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    def retrieve_knowledge_base(
        self,
        knowledge_base_id: str,
        query: str,
        top_k: int,
    ) -> KnowledgeBaseSearchResult:
        if not knowledge_base_id.strip():
            raise ValueError("knowledge_base_id must not be empty")
        if not query.strip():
            raise ValueError("Knowledge Base query must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        started = time.perf_counter()
        response = self.bedrock_agent_runtime.retrieve(
            knowledgeBaseId=knowledge_base_id,
            retrievalQuery={"text": query.strip()},
            retrievalConfiguration={"vectorSearchConfiguration": {"numberOfResults": top_k}},
        )
        documents = []
        for result in response.get("retrievalResults", []):
            content = result.get("content", {})
            documents.append(
                {
                    "text": content.get("text", ""),
                    "score": result.get("score"),
                    "metadata": result.get("metadata", {}),
                    "document_id": result.get("documentId"),
                }
            )
        return KnowledgeBaseSearchResult(
            documents=documents,
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    def sync_knowledge_base(
        self,
        knowledge_base_id: str,
        data_source_id: str,
        poll_seconds: float = 5,
        max_attempts: int = 360,
    ) -> IngestionResult:
        if poll_seconds < 0:
            raise ValueError("poll_seconds must not be negative")
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        response = self.bedrock_agent.start_ingestion_job(
            knowledgeBaseId=knowledge_base_id,
            dataSourceId=data_source_id,
            description="Synchronize the semantic layer benchmark corpus",
        )
        ingestion_job_id = response["ingestionJob"]["ingestionJobId"]
        previous_status = ""
        for _ in range(max_attempts):
            job = self.bedrock_agent.get_ingestion_job(
                knowledgeBaseId=knowledge_base_id,
                dataSourceId=data_source_id,
                ingestionJobId=ingestion_job_id,
            )["ingestionJob"]
            status = job["status"]
            if status != previous_status:
                print(f"Ingestion {knowledge_base_id}/{data_source_id}: {status}")
                previous_status = status
            if status == "COMPLETE":
                return IngestionResult(
                    ingestion_job_id=ingestion_job_id,
                    status=status,
                    statistics={
                        key: int(value) for key, value in job.get("statistics", {}).items()
                    },
                )
            if status in {"FAILED", "STOPPED"}:
                reasons = "; ".join(job.get("failureReasons", [])) or "no reason returned"
                raise ValueError(f"Knowledge Base ingestion {status}: {reasons}")
            time.sleep(poll_seconds)
        raise TimeoutError(
            f"Knowledge Base ingestion {ingestion_job_id} did not finish after "
            f"{max_attempts * poll_seconds:.0f} seconds"
        )

    def purge_vectors(self, bucket: str, index: str) -> int:
        try:
            keys = self._list_vector_keys(bucket, index)
        except ClientError as error:
            if _is_not_found(error):
                return 0
            raise
        self._delete_vector_keys(bucket, index, keys)
        return len(keys)

    def _list_vector_keys(self, bucket: str, index: str) -> list[str]:
        keys: list[str] = []
        paginator = self.s3vectors.get_paginator("list_vectors")
        for page in paginator.paginate(
            vectorBucketName=bucket,
            indexName=index,
            returnData=False,
            returnMetadata=False,
        ):
            keys.extend(vector["key"] for vector in page.get("vectors", []))
        return keys

    def _delete_vector_keys(self, bucket: str, index: str, keys: list[str]) -> None:
        for start in range(0, len(keys), 500):
            self.s3vectors.delete_vectors(
                vectorBucketName=bucket,
                indexName=index,
                keys=keys[start : start + 500],
            )

    def force_delete(self, bucket: str, index: str) -> None:
        self.purge_vectors(bucket, index)
        try:
            self.s3vectors.delete_index(vectorBucketName=bucket, indexName=index)
        except ClientError as error:
            if not _is_not_found(error):
                raise
        for attempt in range(10):
            try:
                self.s3vectors.delete_vector_bucket(vectorBucketName=bucket)
                return
            except ClientError as error:
                if _is_not_found(error):
                    return
                if _is_conflict(error) and attempt < 9:
                    time.sleep(2)
                    continue
                raise

    def vector_bucket_exists(self, bucket: str) -> bool:
        try:
            self.s3vectors.get_vector_bucket(vectorBucketName=bucket)
            return True
        except ClientError as error:
            if _is_not_found(error):
                return False
            raise

    def s3_bucket_exists(self, bucket: str) -> bool:
        try:
            self.s3.head_bucket(Bucket=bucket)
            return True
        except ClientError as error:
            if _is_not_found(error):
                return False
            raise

    def knowledge_base_exists(self, name: str) -> bool:
        paginator = self.bedrock_agent.get_paginator("list_knowledge_bases")
        return any(
            summary.get("name") == name
            for summary in paginator.paginate().search("knowledgeBaseSummaries[]")
            if summary is not None
        )

    def neptune_graph_exists(self, name: str) -> bool:
        paginator = self.neptune_graph.get_paginator("list_graphs")
        return any(
            graph.get("name") == name
            for graph in paginator.paginate().search("graphs[]")
            if graph is not None
        )

    def get_index(self, bucket: str, index: str) -> dict[str, Any]:
        return self.s3vectors.get_index(vectorBucketName=bucket, indexName=index)


def _is_not_found(error: ClientError) -> bool:
    code = str(error.response.get("Error", {}).get("Code", ""))
    status = error.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    return code in {"NotFoundException", "NoSuchVectorBucket", "NoSuchIndex"} or status == 404


def _is_conflict(error: ClientError) -> bool:
    code = str(error.response.get("Error", {}).get("Code", ""))
    status = error.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    return code in {"ConflictException", "ResourceInUseException"} or status == 409
