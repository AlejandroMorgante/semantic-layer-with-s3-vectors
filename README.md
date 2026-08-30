# Building a Semantic Layer with Amazon S3 Vectors

Natural-language query agents need to discover the right tables before they can generate useful
SQL. Sending the complete lakehouse catalog on every request works at small scale, but its token
usage grows with every country, domain, and medallion layer—even when the answer needs one table.

This repository isolates that table-discovery problem and provides a reproducible benchmark for
one narrow question:

> What changes when an AI agent receives an entire data catalog in every prompt versus retrieving
> only the relevant table descriptions from Amazon S3 Vectors?

It does not deploy a lakehouse, execute SQL, or require Amazon Athena. The experiment isolates
context retrieval so the trade-off is easy to measure.

## What we built

The benchmark generates a fully synthetic catalog with 300 table descriptions across Argentina,
Mexico, and the United States. Each table includes its business description, important columns,
country, domain, and bronze, silver, or gold layer.

It then asks the same Amazon Bedrock model the same nine table-selection questions using two paths:

1. **Full Markdown catalog:** load all 300 descriptions into every model request.
2. **Amazon S3 Vectors:** let the agent call `search_tables`, filter candidates by country and
   medallion layer, and return only the Top-K semantic matches.

One table produces one vector. No business data, SQL engine, or real data catalog is required.

## Result at a glance

A run on August 30, 2026 used the default 300-table catalog, nine deterministic questions,
`TOP_K=5`, Amazon Nova Micro, and Amazon Titan Text Embeddings V2 in `us-east-1`:

| Strategy | Accuracy | Mean input tokens | Mean latency | Estimated variable cost/question |
| --- | ---: | ---: | ---: | ---: |
| Full Markdown catalog | 100% | 29,888.3 | 1,391.6 ms | $0.00104814 |
| S3 Vectors retrieval | 100% | 2,094.6 | 2,104.6 ms | $0.00010408 |

In this run, retrieval reduced mean model input by **93.0%** and the included variable-cost estimate
by **90.1%**, with the same 9/9 table-selection accuracy. It added latency because the agent first
generated an embedding, queried S3 Vectors, and completed a second model call.

These are sample measurements, not service guarantees. Run the benchmark in your own account and
Region before using them for an architecture decision.

## What the benchmark compares

The same model answers the same table-selection questions using two strategies:

1. **Full catalog context** loads every table description from one Markdown file into the prompt.
2. **S3 Vectors retrieval** requires the agent to call a `search_tables` tool. The tool filters on
   country and medallion layer before returning the Top-K semantic matches. Domain remains indexed
   metadata but semantic similarity resolves the topic in this benchmark.

Domain remains indexed metadata, while semantic similarity resolves the requested topic.

## Recommended one-command run

Prerequisites:

- Terraform 1.8 or newer
- uv
- AWS CLI v2
- AWS credentials with access to Amazon Bedrock and Amazon S3 Vectors
- An AWS Region that supports both services

Run the complete disposable experiment:

```bash
make demo AWS_PROFILE=my-profile AWS_REGION=us-east-1
```

`make demo` executes `up -> benchmark -> destroy`. Its exit trap calls `destroy` even when indexing
or benchmarking fails.

The default catalog has 100 tables per country, 300 total. Override the scale and repetitions:

```bash
make demo \
  AWS_PROFILE=my-profile \
  TABLES_PER_COUNTRY=300 \
  REPETITIONS=3
```

## Resource lifecycle

Lifecycle safety is a primary design requirement.

### `make up`

```bash
make up AWS_PROFILE=my-profile
```

This command:

1. Validates local tools and AWS credentials.
2. Runs `terraform init`, formatting checks, and validation.
3. Creates a dedicated S3 vector bucket and vector index.
4. Verifies Bedrock model access.
5. Generates the synthetic catalog.
6. Creates embeddings and writes one vector per table.

If any step fails, `up` invokes `destroy` before returning an error. A successful `up` intentionally
leaves resources running so benchmarks can be repeated.

### `make destroy`

```bash
make destroy AWS_PROFILE=my-profile
```

This command:

1. Deletes all vectors from the exact project index.
2. Runs `terraform destroy`.
3. Uses an exact-name AWS API fallback if Terraform deletion fails.
4. Calls `GetVectorBucket` and fails unless absence is verified.

The Terraform vector bucket also sets `force_destroy = true`. Resource names are deterministic and
include the AWS account and Region, allowing cleanup even if local Terraform output is unavailable.

Do not rename `PROJECT_NAME`, switch accounts, or switch Regions between `up` and `destroy`.

## Security and privacy

- The repository contains only synthetic table names, descriptions, columns, and questions.
- AWS credentials are never stored in the project. boto3, the AWS CLI, and Terraform use the
  standard AWS credential chain or the `AWS_PROFILE` supplied at runtime.
- Generated catalogs, benchmark output, Terraform state, variable files, `.env` files, and local
  tool caches are excluded from Git.
- Resource names are derived at deployment time. No AWS account ID or local profile name is
  committed.
- The Terraform configuration creates only a dedicated vector bucket and index. `make demo`
  destroys both and verifies that the vector bucket is absent before it exits.

Review the generated `.benchmark/` artifacts before sharing them if you replace the synthetic
catalog or questions with your own organizational metadata.

## Repeat a benchmark

After `make up`, run:

```bash
make benchmark AWS_PROFILE=my-profile REPETITIONS=3 TOP_K=5
```

Then remove the infrastructure:

```bash
make destroy AWS_PROFILE=my-profile
```

## Outputs

Generated artifacts are written below `.benchmark/`:

```text
.benchmark/
├── catalog.json
├── catalog.md
├── index-metrics.json
└── results/
    ├── comparison.csv
    ├── raw-results.json
    ├── run-metadata.json
    └── summary.md
```

Per-question results include:

- expected, retrieved, and selected tables
- selection accuracy and retrieval recall
- metadata filters chosen by the agent
- model input and output tokens
- query embedding tokens
- retrieval and end-to-end latency
- estimated variable cost

## Cost methodology

Pricing defaults are dated and recorded in every run. The current defaults use:

- Amazon Nova Micro model input and output tokens
- Amazon Titan Text Embeddings V2 input tokens
- the S3 Vectors per-query API charge

The per-question estimate intentionally excludes S3 Vectors storage, PUT, query-processing, and
data-return charges because those depend on logical vector and metadata size. Indexing metrics are
reported separately so one-time ingestion cost is not mixed with per-question retrieval cost.

Before publishing results, verify the rates in `Pricing` against the official AWS pricing pages.

## Infrastructure

Terraform creates only:

- one `aws_s3vectors_vector_bucket`
- one `aws_s3vectors_index`

The vector index uses 256-dimensional normalized Titan embeddings and cosine distance. `country`,
`layer`, and `domain` remain filterable. The full table context is marked non-filterable so it can be
returned to the agent without participating in filter processing.

## Development

```bash
uv sync
make check
```

The AWS profile is never hardcoded. The default profile name is `default`; pass `AWS_PROFILE` on the
command line or export it in your shell.

## License

MIT
