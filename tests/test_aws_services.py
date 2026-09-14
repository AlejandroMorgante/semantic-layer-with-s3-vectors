import pytest
from botocore.exceptions import ClientError

from semantic_layer_benchmark.aws_services import AwsServices, EmbeddingResult
from semantic_layer_benchmark.catalog import generate_catalog


class FakeSession:
    region_name = "us-east-1"

    def client(self, service_name: str, **kwargs: object) -> object:
        return object()


def test_aws_services_leaves_profile_unset_for_default_credential_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_arguments: dict[str, object] = {}

    def fake_session(**kwargs: object) -> FakeSession:
        session_arguments.update(kwargs)
        return FakeSession()

    monkeypatch.delenv("AWS_PROFILE", raising=False)
    monkeypatch.setattr("semantic_layer_benchmark.aws_services.boto3.Session", fake_session)

    AwsServices(region="us-east-1")

    assert session_arguments["profile_name"] is None


class FakeBedrockAgent:
    def __init__(self, jobs: list[dict]) -> None:
        self.jobs = iter(jobs)
        self.started_with: dict | None = None

    def start_ingestion_job(self, **kwargs: object) -> dict:
        self.started_with = kwargs
        return {"ingestionJob": {"ingestionJobId": "job-1"}}

    def get_ingestion_job(self, **kwargs: object) -> dict:
        return {"ingestionJob": next(self.jobs)}


def _services_with_jobs(jobs: list[dict]) -> AwsServices:
    services = object.__new__(AwsServices)
    services.bedrock_agent = FakeBedrockAgent(jobs)
    return services


def test_sync_knowledge_base_waits_until_complete() -> None:
    services = _services_with_jobs(
        [
            {"status": "IN_PROGRESS"},
            {
                "status": "COMPLETE",
                "statistics": {"numberOfNewDocumentsIndexed": 300},
            },
        ]
    )

    result = services.sync_knowledge_base("kb-1", "ds-1", poll_seconds=0)

    assert result.status == "COMPLETE"
    assert result.ingestion_job_id == "job-1"
    assert result.statistics == {"numberOfNewDocumentsIndexed": 300}
    assert services.bedrock_agent.started_with == {
        "knowledgeBaseId": "kb-1",
        "dataSourceId": "ds-1",
        "description": "Synchronize the semantic layer benchmark corpus",
    }


def test_sync_knowledge_base_surfaces_failure_reason() -> None:
    services = _services_with_jobs(
        [{"status": "FAILED", "failureReasons": ["model access denied"]}]
    )

    with pytest.raises(ValueError, match="model access denied"):
        services.sync_knowledge_base("kb-1", "ds-1", poll_seconds=0)


class FakePaginator:
    def __init__(self, pages: list[dict]) -> None:
        self.pages = pages

    def paginate(self, **kwargs: object) -> list[dict]:
        return self.pages


class FakeS3Vectors:
    def __init__(self) -> None:
        self.puts: list[dict] = []
        self.deletes: list[dict] = []

    def get_paginator(self, name: str) -> FakePaginator:
        assert name == "list_vectors"
        return FakePaginator(
            [{"vectors": [{"key": "argentina_gold_sales_daily"}, {"key": "stale"}]}]
        )

    def put_vectors(self, **kwargs: object) -> None:
        self.puts.append(kwargs)

    def delete_vectors(self, **kwargs: object) -> None:
        self.deletes.append(kwargs)


def test_index_catalog_deletes_keys_missing_from_current_catalog() -> None:
    services = object.__new__(AwsServices)
    services.s3vectors = FakeS3Vectors()
    services.embed = lambda text, model_id: EmbeddingResult([0.1], 3)  # type: ignore[method-assign]
    record = next(
        record for record in generate_catalog(10) if record.key == "argentina_gold_sales_daily"
    )

    metrics = services.index_catalog("bucket", "index", [record], "embedding", workers=1)

    assert metrics["vectors_indexed"] == 1
    assert metrics["vectors_deleted"] == 1
    assert services.s3vectors.deletes == [
        {"vectorBucketName": "bucket", "indexName": "index", "keys": ["stale"]}
    ]


class FakeS3:
    def __init__(self, error: ClientError) -> None:
        self.error = error

    def head_bucket(self, **kwargs: object) -> None:
        raise self.error


def _client_error(code: str, status: int) -> ClientError:
    return ClientError(
        {
            "Error": {"Code": code, "Message": code},
            "ResponseMetadata": {"HTTPStatusCode": status},
        },
        "HeadBucket",
    )


def test_s3_bucket_exists_returns_false_only_for_not_found() -> None:
    services = object.__new__(AwsServices)
    services.s3 = FakeS3(_client_error("NoSuchBucket", 404))

    assert services.s3_bucket_exists("missing") is False


def test_s3_bucket_exists_propagates_access_denied() -> None:
    services = object.__new__(AwsServices)
    error = _client_error("AccessDenied", 403)
    services.s3 = FakeS3(error)

    with pytest.raises(ClientError) as raised:
        services.s3_bucket_exists("private")

    assert raised.value is error
