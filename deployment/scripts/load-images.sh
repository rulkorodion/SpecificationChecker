#!/usr/bin/env bash
set -Eeuo pipefail

# shellcheck source=common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
load_environment
validate_environment
require_command docker
require_command sha256sum

archive_name="specification-checker-images-$VERSION-$(platform_slug).tar"
archive_path="$DEPLOY_ROOT/docker-images/$archive_name"
[[ -s "$archive_path" ]] || die "Docker image archive is missing or empty: $archive_path"

checksum_line="$(grep -F "  docker-images/$archive_name" "$DEPLOY_ROOT/checksums.sha256" || true)"
[[ -n "$checksum_line" ]] || die "Archive checksum is absent from checksums.sha256"
(cd "$DEPLOY_ROOT" && printf '%s\n' "$checksum_line" | sha256sum --check --strict -)

docker load --input "$archive_path"
for image in "$API_IMAGE" "$WORKER_IMAGE" "$POSTGRES_IMAGE" "$REDIS_IMAGE"; do
  image_exists "$image" || die "Expected tag was not loaded: $image"
done

note "All offline images are loaded and verified. No pull was used."
