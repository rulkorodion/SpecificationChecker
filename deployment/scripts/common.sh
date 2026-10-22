#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="${ENV_FILE:-$DEPLOY_ROOT/.env.production}"
COMPOSE_FILE="$DEPLOY_ROOT/compose.production.yml"
VERSION="$(tr -d '\r\n' < "$DEPLOY_ROOT/VERSION")"
API_IMAGE_VERSION="$(tr -d '\r\n' < "$DEPLOY_ROOT/API_IMAGE_VERSION")"
WORKER_IMAGE_VERSION="$(tr -d '\r\n' < "$DEPLOY_ROOT/WORKER_IMAGE_VERSION")"
API_IMAGE="specification-checker-api:$API_IMAGE_VERSION"
WORKER_IMAGE="specification-checker-worker:$WORKER_IMAGE_VERSION"
POSTGRES_IMAGE="postgres:18.4-alpine"
REDIS_IMAGE="redis:7.4.10-alpine"

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

note() {
  printf '%s\n' "$*"
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "Required command is missing: $1"
}

load_environment() {
  [[ -f "$ENV_FILE" ]] || die "Create $ENV_FILE from .env.production.example"
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
}

require_variable() {
  local name="$1"
  [[ -n "${!name:-}" ]] || die "Required variable is empty: $name"
}

validate_environment() {
  local required=(
    TARGET_PLATFORM POSTGRES_USER POSTGRES_PASSWORD POSTGRES_DB DATABASE_URL
    REDIS_URL REDIS_QUEUE_NAME JOB_STORAGE_ROOT MONKEYOCR_MODEL_PATH
    API_BIND_ADDRESS API_PORT JOB_RETENTION_DAYS ONEC_INTEGRATION_ENABLED
    ONEC_ADAPTER WORKER_JOB_TIMEOUT_SECONDS WORKER_LOCK_TTL_SECONDS
    EXPECTED_WORKER_COUNT MONKEYOCR_DTYPE MONKEYOCR_MAX_PIXELS
    MONKEYOCR_PDF_RENDER_DPI
  )
  local name
  for name in "${required[@]}"; do
    require_variable "$name"
  done
  if [[ "$POSTGRES_PASSWORD" =~ REPLACE|CHANGE|EXAMPLE|password ]]; then
    die "POSTGRES_PASSWORD is still a placeholder"
  fi
  if [[ "$DATABASE_URL" =~ REPLACE|CHANGE|EXAMPLE ]]; then
    die "DATABASE_URL is still a placeholder"
  fi
  [[ "$ONEC_INTEGRATION_ENABLED" =~ ^(true|false)$ ]] || die "Invalid ONEC_INTEGRATION_ENABLED"
  [[ "$ONEC_ADAPTER" =~ ^(mock|http|onec_http)$ ]] || die "Invalid ONEC_ADAPTER"
  if [[ "$ONEC_INTEGRATION_ENABLED" == "true" && "$ONEC_ADAPTER" != "mock" ]]; then
    [[ "$ONEC_API_URL" =~ ^https?:// ]] || die "ONEC_API_URL must be a full http/https URL"
  fi
  [[ "$MONKEYOCR_DTYPE" =~ ^(auto|bfloat16|float16)$ ]] || die "Invalid MONKEYOCR_DTYPE"
  [[ "$MONKEYOCR_MAX_PIXELS" =~ ^[1-9][0-9]*$ ]] || die "Invalid MONKEYOCR_MAX_PIXELS"
  [[ "$MONKEYOCR_PDF_RENDER_DPI" =~ ^[1-9][0-9]*$ ]] || die "Invalid MONKEYOCR_PDF_RENDER_DPI"
  [[ "$EXPECTED_WORKER_COUNT" =~ ^[1-9][0-9]*$ ]] || die "Invalid EXPECTED_WORKER_COUNT"
  (( EXPECTED_WORKER_COUNT == 2 )) || die "This deployment is fixed to exactly two GPU workers"
  [[ -z "${MONKEYOCR_MAX_NEW_TOKENS:-}" || "$MONKEYOCR_MAX_NEW_TOKENS" =~ ^[1-9][0-9]*$ ]] || die "Invalid MONKEYOCR_MAX_NEW_TOKENS"
}

compose() {
  docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" "$@"
}

image_exists() {
  docker image inspect "$1" >/dev/null 2>&1
}

platform_slug() {
  printf '%s' "${TARGET_PLATFORM//\//-}"
}
