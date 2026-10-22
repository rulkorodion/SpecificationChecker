#!/usr/bin/env bash
set -Eeuo pipefail

# shellcheck source=common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
load_environment
validate_environment

require_command uname
require_command docker
require_command nvidia-smi
require_command sha256sum
require_command curl
require_command find
require_command df

[[ "$(uname -s)" == "Linux" ]] || die "This package must be deployed on Linux"
case "$(uname -m)" in
  x86_64|amd64) host_platform="linux/amd64" ;;
  aarch64|arm64) host_platform="linux/arm64" ;;
  *) die "Unsupported server architecture: $(uname -m)" ;;
esac
[[ "$host_platform" == "$TARGET_PLATFORM" ]] || die "Package platform $TARGET_PLATFORM does not match host $host_platform"

docker info >/dev/null 2>&1 || die "Docker Engine is unavailable"
docker compose version >/dev/null 2>&1 || die "Docker Compose plugin is unavailable"
nvidia-smi >/dev/null 2>&1 || die "NVIDIA GPU or driver is unavailable on the host"
gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
(( gpu_count >= EXPECTED_WORKER_COUNT )) \
  || die "Detected GPUs: $gpu_count, workers requested: $EXPECTED_WORKER_COUNT"

[[ -d "$MONKEYOCR_MODEL_PATH" ]] || die "MonkeyOCR model directory does not exist: $MONKEYOCR_MODEL_PATH"
find "$MONKEYOCR_MODEL_PATH" -mindepth 1 -print -quit | grep -q . || die "MonkeyOCR model directory is empty"
model_snapshots="$MONKEYOCR_MODEL_PATH/hub/models--zenosai--MonkeyOCRv2-B-Parsing/snapshots"
[[ -d "$model_snapshots" ]] || die "Expected Hugging Face snapshots directory is missing: $model_snapshots"
find "$model_snapshots" -mindepth 1 -maxdepth 1 -type d -print -quit | grep -q . || die "MonkeyOCR snapshot is missing"

free_kb="$(df -Pk "$DEPLOY_ROOT" | awk 'NR==2 {print $4}')"
free_gb=$((free_kb / 1024 / 1024))
min_free_gb="${MIN_FREE_DISK_GB:-25}"
(( free_gb >= min_free_gb )) || die "Only ${free_gb} GiB is free; ${min_free_gb} GiB is required"

config_output="$(compose config)"
if grep -Eq '^[[:space:]]*build:' <<<"$config_output"; then
  die "Production Compose unexpectedly contains build:"
fi
grep -q 'read_only: true' "$COMPOSE_FILE" || die "MonkeyOCR mount is not read-only"
grep -q 'HF_HUB_OFFLINE: "1"' "$COMPOSE_FILE" || die "HF_HUB_OFFLINE is not enforced"
grep -q 'TRANSFORMERS_OFFLINE: "1"' "$COMPOSE_FILE" || die "TRANSFORMERS_OFFLINE is not enforced"

if image_exists "$WORKER_IMAGE"; then
  docker run --rm --pull never --platform "$TARGET_PLATFORM" --gpus all \
    --entrypoint nvidia-smi "$WORKER_IMAGE" >/dev/null \
    || die "NVIDIA Container Toolkit cannot expose the GPU to the worker image"
  note "Container GPU check: OK"
elif [[ "${REQUIRE_CONTAINER_GPU_CHECK:-false}" == "true" ]]; then
  die "Worker image is not loaded; run load-images.sh first"
else
  note "Container GPU check: deferred until load-images.sh (host GPU is OK)"
fi

note "Host check: OK"
note "Platform: $host_platform"
note "NVIDIA GPUs: $gpu_count"
note "Free disk: ${free_gb} GiB"
note "Model: external read-only Hugging Face cache"
