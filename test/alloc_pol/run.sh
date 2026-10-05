#!/usr/bin/env bash
# Heap allocations, type stability and GPU-readiness of the polarized transfer routines.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ../common.sh

ensure_julia

rm -rf output
mkdir -p output

log "Checking allocations and type stability of the polarized routines"
# One thread: the measurements are of single calls.
$JULIA --project="$REPO_ROOT/scripts" --threads=1 run.jl > output/alloc.log 2>&1 || {
    cat output/alloc.log >&2
    exit 1
}
cat output/alloc.log
