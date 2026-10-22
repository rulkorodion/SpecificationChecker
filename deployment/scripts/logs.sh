#!/usr/bin/env bash
set -Eeuo pipefail

# shellcheck source=common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
load_environment
validate_environment

service="${1:-}"
case "$service" in
  "") compose logs --tail "${LOG_TAIL:-200}" api worker worker-gpu-1 postgres redis migrate ;;
  api|worker|worker-gpu-1|postgres|redis|migrate) compose logs --tail "${LOG_TAIL:-200}" "$service" ;;
  workers) compose logs --tail "${LOG_TAIL:-200}" worker worker-gpu-1 ;;
  *) die "Allowed services: api, workers, worker, worker-gpu-1, postgres, redis, migrate" ;;
esac
