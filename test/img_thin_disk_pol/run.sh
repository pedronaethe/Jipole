#!/usr/bin/env bash
# Polarized thin disk: ipole (MODEL=thin_disk) against Jipole, Stokes I, Q, U, V.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ../common.sh

#ensure we have ipole (built with the thin disk model), python with necessary libraries and julia
ensure_ipole thin_disk
ensure_python
ensure_julia

#redirect logs and images to output folder
rm -rf output
mkdir -p output

# ipole reads Chandrasekhar's table from the directory it is run in.
cp "$IPOLE_DIR/model/thin_disk/ch24_vals.txt" output/

log "Running ipole (thin disk, polarized)"
(cd output && OMP_NUM_THREADS="${OMP_NUM_THREADS:-$NPROC}" "$IPOLE_BIN" -par ../ipole.par > ipole.log 2>&1) || {
    tail -20 output/ipole.log >&2
    log "ipole failed"
    exit 1
}

log "Running Jipole (thin disk, polarized)"
$JULIA --project="$REPO_ROOT/scripts" --threads="$NPROC" run.jl jipole.toml > output/jipole.log 2>&1 || {
    tail -20 output/jipole.log >&2
    log "Jipole failed"
    exit 1
}

"$PYTHON" check.py output/jipole.h5 output/ipole.h5 output/comparison.png
