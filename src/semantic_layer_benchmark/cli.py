from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from botocore.exceptions import ClientError

from semantic_layer_benchmark.aws_services import AwsServices
from semantic_layer_benchmark.benchmark import run_benchmark
from semantic_layer_benchmark.catalog import generate_catalog, read_catalog, write_catalog


def command_generate(args: argparse.Namespace) -> None:
    records = generate_catalog(args.tables_per_country)
    json_path, markdown_path = write_catalog(records, args.output_dir)
    print(f"Generated {len(records)} tables: {json_path} and {markdown_path}")


def command_preflight(args: argparse.Namespace) -> None:
    services = AwsServices()
    identity = services.identity()
    services.get_index(args.bucket, args.index)
    services.embed("semantic retrieval preflight", args.embedding_model_id)
    response = services.bedrock.converse(
        modelId=args.model_id,
        messages=[{"role": "user", "content": [{"text": "Reply with the word ready."}]}],
        inferenceConfig={"maxTokens": 16, "temperature": 0},
    )
    if not response.get("output"):
        raise ValueError("Bedrock model preflight returned no output")
    print(
        f"Preflight passed for account {identity['Account']} in {services.region}; "
        f"bucket={args.bucket} index={args.index}"
    )


def command_index(args: argparse.Namespace) -> None:
    services = AwsServices()
    metrics = services.index_catalog(
        args.bucket,
        args.index,
        read_catalog(args.catalog),
        args.embedding_model_id,
    )
    (args.catalog.parent / "index-metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))


def command_benchmark(args: argparse.Namespace) -> None:
    services = AwsServices()
    results = run_benchmark(
        services=services,
        bucket=args.bucket,
        index=args.index,
        catalog_markdown=args.catalog.with_suffix(".md").read_text(encoding="utf-8"),
        questions=json.loads(args.questions.read_text(encoding="utf-8")),
        model_id=args.model_id,
        embedding_model_id=args.embedding_model_id,
        top_k=args.top_k,
        repetitions=args.repetitions,
        output_dir=args.output_dir,
        s3_vectors_kb_id=args.s3_vectors_kb_id,
        neptune_kb_id=args.neptune_kb_id,
        strategies=args.strategies,
    )
    print(f"Wrote {len(results)} benchmark records to {args.output_dir}")


def command_sync(args: argparse.Namespace) -> None:
    result = AwsServices().sync_knowledge_base(
        knowledge_base_id=args.knowledge_base_id,
        data_source_id=args.data_source_id,
        poll_seconds=args.poll_seconds,
        max_attempts=args.max_attempts,
    )
    print(
        json.dumps(
            {
                "ingestion_job_id": result.ingestion_job_id,
                "status": result.status,
                "statistics": result.statistics,
            },
            indent=2,
        )
    )


def command_purge(args: argparse.Namespace) -> None:
    deleted = AwsServices().purge_vectors(args.bucket, args.index)
    print(f"Deleted {deleted} vectors from {args.bucket}/{args.index}")


def command_force_delete(args: argparse.Namespace) -> None:
    AwsServices().force_delete(args.bucket, args.index)
    print(f"Fallback cleanup completed for {args.bucket}/{args.index}")


def command_verify_destroyed(args: argparse.Namespace) -> None:
    if AwsServices().vector_bucket_exists(args.bucket):
        raise ValueError(f"Vector bucket still exists after destroy: {args.bucket}")
    print(f"Verified absent: {args.bucket}")


def _add_resources(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--index", required=True)


def _parse_strategies(value: str) -> tuple[str, ...]:
    strategies = tuple(item.strip() for item in value.split(",") if item.strip())
    if not strategies:
        raise argparse.ArgumentTypeError("Provide at least one comma-separated strategy")
    return strategies


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark semantic table retrieval")
    subparsers = parser.add_subparsers(required=True)

    generate = subparsers.add_parser("generate")
    generate.add_argument("--tables-per-country", type=int, default=100)
    generate.add_argument("--output-dir", type=Path, required=True)
    generate.set_defaults(handler=command_generate)

    preflight = subparsers.add_parser("preflight")
    _add_resources(preflight)
    preflight.add_argument("--model-id", required=True)
    preflight.add_argument("--embedding-model-id", required=True)
    preflight.set_defaults(handler=command_preflight)

    index = subparsers.add_parser("index")
    _add_resources(index)
    index.add_argument("--catalog", type=Path, required=True)
    index.add_argument("--embedding-model-id", required=True)
    index.set_defaults(handler=command_index)

    benchmark = subparsers.add_parser("benchmark")
    benchmark.add_argument("--bucket")
    benchmark.add_argument("--index")
    benchmark.add_argument("--catalog", type=Path, required=True)
    benchmark.add_argument("--questions", type=Path, required=True)
    benchmark.add_argument("--output-dir", type=Path, required=True)
    benchmark.add_argument("--model-id", required=True)
    benchmark.add_argument("--embedding-model-id")
    benchmark.add_argument(
        "--s3-vectors-kb-id",
        default=os.getenv("S3_VECTORS_KB_ID") or None,
    )
    benchmark.add_argument(
        "--neptune-kb-id",
        default=os.getenv("NEPTUNE_KB_ID") or None,
    )
    benchmark.add_argument(
        "--strategies",
        type=_parse_strategies,
        default=(
            _parse_strategies(os.environ["BENCHMARK_STRATEGIES"])
            if os.getenv("BENCHMARK_STRATEGIES")
            else None
        ),
        help="Comma-separated strategies; inferred from the supplied resources when omitted.",
    )
    benchmark.add_argument("--top-k", type=int, default=5)
    benchmark.add_argument("--repetitions", type=int, default=1)
    benchmark.set_defaults(handler=command_benchmark)

    sync = subparsers.add_parser("sync-knowledge-base")
    sync.add_argument("--knowledge-base-id", required=True)
    sync.add_argument("--data-source-id", required=True)
    sync.add_argument("--poll-seconds", type=float, default=5)
    sync.add_argument("--max-attempts", type=int, default=360)
    sync.set_defaults(handler=command_sync)

    purge = subparsers.add_parser("purge")
    _add_resources(purge)
    purge.set_defaults(handler=command_purge)

    force_delete = subparsers.add_parser("force-delete")
    _add_resources(force_delete)
    force_delete.set_defaults(handler=command_force_delete)

    verify = subparsers.add_parser("verify-destroyed")
    verify.add_argument("--bucket", required=True)
    verify.set_defaults(handler=command_verify_destroyed)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        args.handler(args)
        return 0
    except ClientError as error:
        print(f"AWS error: {error}", file=sys.stderr)
        return 1
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
