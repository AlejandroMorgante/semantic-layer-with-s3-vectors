import pytest

from semantic_layer_benchmark.aws_services import AwsServices


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
