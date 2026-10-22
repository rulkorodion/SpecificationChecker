#!/usr/bin/env bash
set -Eeuo pipefail

# shellcheck source=common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
load_environment
validate_environment

compose down
note "Containers stopped. Named volumes and all persisted data were preserved."
