#!/usr/bin/env bash
set -euo pipefail
component=${1:?Usage: bash run_continuation.sh a_or_b}
case "$component" in a|b) ;; *) exit 2 ;; esac
cd "$(dirname "$0")/alpha_cen_${component}"
export MESA_CACHES_DIR="$PWD/.mesa_caches"
export MESA_TEMP_CACHES_DIR="$PWD/.mesa_temp_cache"
export OMP_NUM_THREADS=4
mkdir -p "$MESA_CACHES_DIR" "$MESA_TEMP_CACHES_DIR"
make > build.log 2>&1
test ! -e run.log
{
    date -u '+START UTC: %Y-%m-%dT%H:%M:%SZ'
    /usr/bin/time -p ./build/bin/star
    date -u '+END UTC: %Y-%m-%dT%H:%M:%SZ'
} > run.log 2>&1
