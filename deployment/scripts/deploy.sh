#!/usr/bin/env bash
set -Eeuo pipefail

# shellcheck source=common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
load_environment
validate_environment

for image in "$API_IMAGE" "$WORKER_IMAGE" "$POSTGRES_IMAGE" "$REDIS_IMAGE"; do
  image_exists "$image" || die "Image is not loaded: $image. Run scripts/load-images.sh"
done

REQUIRE_CONTAINER_GPU_CHECK=true "$SCRIPT_DIR/check-host.sh"

note "Starting PostgreSQL and Redis without build or pull..."
compose up -d --no-build --pull never postgres redis

note "Running Alembic migrations..."
compose rm -f -s migrate >/dev/null 2>&1 || true
compose up --no-build --pull never migrate

note "Starting API and two GPU workers (GPU 0 and GPU 1)..."
compose up -d --no-build --pull never api worker worker-gpu-1

deadline=$((SECONDS + 240))
until curl --silent --show-error --fail "${API_BASE_URL:-http://127.0.0.1:${API_PORT}}/health" >/dev/null 2>&1; do
  (( SECONDS < deadline )) || {
    compose ps --all
    die "API/workers health did not become ready within 240 seconds"
  }
  sleep 3
done

"$SCRIPT_DIR/status.sh"
note "Deployment is healthy. PostgreSQL/job volumes were preserved."
