#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

preflight_tools
uv sync --project "${ROOT_DIR}" --frozen
tf init -input=false

account_id="$(aws sts get-caller-identity --profile "${AWS_PROFILE}" --query Account --output text)"
fallback_bucket="${PROJECT_NAME}-${account_id}-${AWS_REGION}"
fallback_bucket="${fallback_bucket:0:63}"
bucket_name="$(tf output -raw vector_bucket_name 2>/dev/null || echo "${fallback_bucket}")"
index_name="$(tf output -raw vector_index_name 2>/dev/null || echo "tables")"

# Empty the exact project index first. Terraform force_destroy is the second
# cleanup layer and handles interrupted indexing runs.
benchmark_cli purge --bucket "${bucket_name}" --index "${index_name}" || true

if ! tf destroy -auto-approve -input=false; then
  echo "Terraform destroy failed. Attempting exact-name fallback cleanup." >&2
  benchmark_cli force-delete --bucket "${bucket_name}" --index "${index_name}"
  tf destroy -auto-approve -input=false
fi

benchmark_cli verify-destroyed --bucket "${bucket_name}"
echo "DESTROY complete. Verified that vector bucket '${bucket_name}' does not exist."
