#!/usr/bin/env bash
# Polarized GPU kernel: its per-pixel body, run on the CPU, against the CPU path.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ../common.sh

ensure_julia

rm -rf output
mkdir -p output

log "Running the polarized GPU kernel body on the CPU"
$JULIA --project="$REPO_ROOT/scripts" --threads=1 run.jl > output/gpu_pol.log 2>&1 || {
    cat output/gpu_pol.log >&2
    exit 1
}
cat output/gpu_pol.log
