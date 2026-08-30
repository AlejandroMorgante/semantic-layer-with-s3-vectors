#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TF_DIR="${ROOT_DIR}/terraform"
BENCHMARK_DIR="${ROOT_DIR}/.benchmark"

AWS_PROFILE="${AWS_PROFILE:-default}"
AWS_REGION="${AWS_REGION:-us-east-1}"
PROJECT_NAME="${PROJECT_NAME:-semantic-layer-benchmark}"
TABLES_PER_COUNTRY="${TABLES_PER_COUNTRY:-100}"
REPETITIONS="${REPETITIONS:-1}"
TOP_K="${TOP_K:-5}"
MODEL_ID="${MODEL_ID:-us.amazon.nova-micro-v1:0}"
EMBEDDING_MODEL_ID="${EMBEDDING_MODEL_ID:-amazon.titan-embed-text-v2:0}"

export AWS_PROFILE AWS_REGION AWS_DEFAULT_REGION="${AWS_REGION}"
export TF_VAR_aws_region="${AWS_REGION}" TF_VAR_project_name="${PROJECT_NAME}"

tf() {
  terraform -chdir="${TF_DIR}" "$@"
}
benchmark_cli() {
  uv run --project "${ROOT_DIR}" semantic-layer-benchmark "$@"
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
  aws sts get-caller-identity --profile "${AWS_PROFILE}" --region "${AWS_REGION}" >/dev/null
}

terraform_output() {
  tf output -raw "$1"
}
