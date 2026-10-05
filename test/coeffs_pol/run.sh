#!/usr/bin/env bash
# Polarized thermal synchrotron coefficients against ipole's own fit routines (stored reference).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ../common.sh

ensure_julia

rm -rf output
mkdir -p output

log "Checking the polarized transfer coefficients"
$JULIA --project="$REPO_ROOT/scripts" run.jl > output/coeffs.log 2>&1 || {
    cat output/coeffs.log >&2
    exit 1
}
cat output/coeffs.log
