AWS_PROFILE ?= default
AWS_REGION ?= us-east-1
PROJECT_NAME ?= semantic-layer-benchmark
TABLES_PER_COUNTRY ?= 100
REPETITIONS ?= 1
TOP_K ?= 5
MODEL_ID ?= us.amazon.nova-micro-v1:0
EMBEDDING_MODEL_ID ?= amazon.titan-embed-text-v2:0
GRAPH_CONSTRUCTION_MODEL_ID ?= amazon.nova-micro-v1:0
NEPTUNE_PROVISIONED_MEMORY ?= 16
S3_VECTORS_KB_ID ?=
NEPTUNE_KB_ID ?=
BENCHMARK_STRATEGIES ?=

export AWS_PROFILE
export AWS_REGION
export AWS_DEFAULT_REGION := $(AWS_REGION)
export PROJECT_NAME
export TABLES_PER_COUNTRY
export REPETITIONS
export TOP_K
export MODEL_ID
export EMBEDDING_MODEL_ID
export GRAPH_CONSTRUCTION_MODEL_ID
export NEPTUNE_PROVISIONED_MEMORY
export S3_VECTORS_KB_ID
export NEPTUNE_KB_ID
export BENCHMARK_STRATEGIES

.PHONY: help generate up benchmark destroy demo test lint fmt check

help:
	@echo "make demo       Deploy, benchmark, and always destroy (recommended)"
	@echo "make generate   Generate the shared Markdown and Knowledge Base corpus"
	@echo "make up         Deploy and index; resources remain until make destroy"
	@echo "make benchmark  Run against resources created by make up"
	@echo "                Add both KB IDs to run the four-strategy benchmark"
	@echo "make destroy    Remove vectors, index, and vector bucket; verify removal"
	@echo "make check      Run formatting, lint, tests, and Terraform validation"

up:
	@./scripts/up.sh

generate:
	@uv run semantic-layer-benchmark generate \
		--tables-per-country "$(TABLES_PER_COUNTRY)" \
		--output-dir .benchmark

benchmark:
	@./scripts/benchmark.sh

destroy:
	@./scripts/destroy.sh

demo:
	@./scripts/demo.sh

test:
	@uv run pytest

lint:
	@uv run ruff check .

fmt:
	@uv run ruff format .
	@terraform -chdir=terraform fmt

check:
	@uv run ruff format --check .
	@uv run ruff check .
	@uv run pytest
	@terraform -chdir=terraform fmt -check
	@terraform -chdir=terraform init -backend=false -input=false
	@terraform -chdir=terraform validate
