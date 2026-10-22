#!/usr/bin/env bash
set -Eeuo pipefail

# shellcheck source=common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
load_environment
validate_environment
require_command curl

compose ps --all
health="$(curl --silent --show-error --fail "${API_BASE_URL:-http://127.0.0.1:${API_PORT}}/health")" \
  || die "API health endpoint is unavailable"
printf '%s\n' "$health"
grep -q '"status":"ok"' <<<"$health" || die "Health status is not ok"
grep -q '"database_available":true' <<<"$health" || die "PostgreSQL is unavailable"
grep -q '"redis_available":true' <<<"$health" || die "Redis is unavailable"
grep -q '"worker_registered":true' <<<"$health" || die "Worker is not registered"
worker_count="$(python3 -c 'import json,sys; print(json.load(sys.stdin).get("worker_count", 0))' <<<"$health")"
(( worker_count >= EXPECTED_WORKER_COUNT )) \
  || die "Registered workers: $worker_count, expected: $EXPECTED_WORKER_COUNT"
note "Registered workers: $worker_count"
note "Service status: OK"
