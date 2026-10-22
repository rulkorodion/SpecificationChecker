#!/usr/bin/env bash
set -Eeuo pipefail

# shellcheck source=common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
load_environment
validate_environment
require_command curl
require_command python3

base_url="${API_BASE_URL:-http://127.0.0.1:${API_PORT}}"
document="$DEPLOY_ROOT/smoke/document.pdf"
reference="$DEPLOY_ROOT/smoke/reference.json"
[[ -s "$document" ]] || die "Smoke PDF is missing"
[[ -s "$reference" ]] || die "Smoke reference is missing"

health="$(curl --silent --show-error --fail "$base_url/health")" || die "API is unavailable"
grep -q '"worker_registered":true' <<<"$health" || die "Worker is not registered; a real worker run cannot be proven"

response_file="$(mktemp)"
trap 'rm -f "$response_file"' EXIT
started_epoch="$(date +%s)"
http_code="$(curl --silent --show-error --output "$response_file" --write-out '%{http_code}' \
  --form "document=@$document;type=application/pdf" \
  --form "reference=<$reference" \
  --form "external_id=deployment-smoke-$started_epoch" \
  "$base_url/api/v1/checks")"
response="$(cat "$response_file")"
[[ "$http_code" == "202" ]] || {
  printf '%s\n' "$response" >&2
  die "POST returned HTTP $http_code instead of 202"
}
job_id="$(printf '%s' "$response" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("job_id", ""))')"
[[ -n "$job_id" ]] || die "POST response has no job_id"
post_elapsed=$(( $(date +%s) - started_epoch ))
note "POST: HTTP 202, job_id=$job_id, response_time=${post_elapsed}s"

timeout="${SMOKE_TIMEOUT_SECONDS:-3600}"
deadline=$(( $(date +%s) + timeout ))
while :; do
  response="$(curl --silent --show-error --fail "$base_url/api/v1/checks/$job_id")" \
    || die "GET failed for job_id=$job_id"
  status="$(printf '%s' "$response" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("status", ""))')"
  case "$status" in
    COMPLETED)
      decision="$(printf '%s' "$response" | python3 -c 'import json,sys; data=json.load(sys.stdin); print((data.get("result") or {}).get("decision", ""))')"
      processing="$(printf '%s' "$response" | python3 -c 'import json,sys; value=json.load(sys.stdin).get("processing_time_seconds"); print("" if value is None else value)')"
      [[ -n "$decision" ]] || die "Completed result has no decision"
      [[ -n "$processing" && "$processing" != "null" ]] || die "Completed result has no processing time"
      note "Smoke test: COMPLETED"
      note "Decision: $decision"
      note "Processing time: ${processing}s"
      note "Total elapsed: $(( $(date +%s) - started_epoch ))s"
      break
      ;;
    FAILED)
      printf '%s\n' "$response" >&2
      die "Smoke job FAILED"
      ;;
    PROCESSING) ;;
    *) die "Unexpected job status: $status" ;;
  esac
  (( $(date +%s) < deadline )) || die "Smoke job timed out after ${timeout}s"
  sleep 5
done
