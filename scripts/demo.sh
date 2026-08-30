#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

always_destroy() {
  status=$?
  echo "Running mandatory cleanup."
  "${ROOT_DIR}/scripts/destroy.sh" || {
    echo "Cleanup failed. Run make destroy immediately." >&2
    exit 1
  }
  exit "${status}"
}
trap always_destroy EXIT

"${ROOT_DIR}/scripts/up.sh"
"${ROOT_DIR}/scripts/benchmark.sh"
