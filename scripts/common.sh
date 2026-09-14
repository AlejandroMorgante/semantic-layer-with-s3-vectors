#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TF_DIR="${ROOT_DIR}/terraform"
BENCHMARK_DIR="${ROOT_DIR}/.benchmark"

AWS_PROFILE="${AWS_PROFILE:-}"
AWS_REGION="${AWS_REGION:-us-east-1}"
PROJECT_NAME="${PROJECT_NAME:-semantic-layer-benchmark}"
TABLES_PER_COUNTRY="${TABLES_PER_COUNTRY:-100}"
REPETITIONS="${REPETITIONS:-1}"
TOP_K="${TOP_K:-5}"
MODEL_ID="${MODEL_ID:-us.amazon.nova-micro-v1:0}"
EMBEDDING_MODEL_ID="${EMBEDDING_MODEL_ID:-amazon.titan-embed-text-v2:0}"
GRAPH_CONSTRUCTION_MODEL_ID="${GRAPH_CONSTRUCTION_MODEL_ID:-amazon.nova-micro-v1:0}"
NEPTUNE_PROVISIONED_MEMORY="${NEPTUNE_PROVISIONED_MEMORY:-16}"
S3_VECTORS_KB_ID="${S3_VECTORS_KB_ID:-}"
NEPTUNE_KB_ID="${NEPTUNE_KB_ID:-}"
BENCHMARK_STRATEGIES="${BENCHMARK_STRATEGIES:-}"

if [[ -n "${AWS_PROFILE}" ]]; then
  export AWS_PROFILE
else
  unset AWS_PROFILE
fi
export AWS_REGION AWS_DEFAULT_REGION="${AWS_REGION}"
export TF_VAR_aws_region="${AWS_REGION}" TF_VAR_project_name="${PROJECT_NAME}"
export TF_VAR_embedding_model_id="${EMBEDDING_MODEL_ID}"
export TF_VAR_graph_construction_model_id="${GRAPH_CONSTRUCTION_MODEL_ID}"
export TF_VAR_neptune_provisioned_memory="${NEPTUNE_PROVISIONED_MEMORY}"

tf() {
  terraform -chdir="${TF_DIR}" "$@"
}
benchmark_cli() {
  uv run --project "${ROOT_DIR}" semantic-layer-benchmark "$@"
}
aws_cli() {
  local aws_options=(--region "${AWS_REGION}")
  if [[ -n "${AWS_PROFILE:-}" ]]; then
    aws_options+=(--profile "${AWS_PROFILE}")
  fi
  aws "${aws_options[@]}" "$@"
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Required command not found: $1" >&2
    return 1
  fi
}

preflight_tools() {
  require_command aws
  require_command terraform
  require_command uv
  aws_cli sts get-caller-identity >/dev/null
}

terraform_output() {
  tf output -raw "$1"
}
