#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ../common.sh

#ensure we have ipole, python with necessary libraries and julia
#these functions are defined in common.sh
ensure_ipole iharm
ensure_python
ensure_julia
dump="$(ensure_file sample_dump_SANE_a+0.94_MKS_0900.h5 \
    https://dataverse.harvard.edu/api/access/datafile/12137142 dc8cf9b45136cd9b28bdb8ed197cfd69)"


#redirect logs and images to output folder
rm -rf output
mkdir -p output
ln -s "$dump" output/dump.h5

log "Running ipole"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-$NPROC}" "$IPOLE_BIN" -par ipole.par > output/ipole.log 2>&1 || {
    tail -20 output/ipole.log >&2
    log "ipole failed"
    exit 1
}

log "Running Jipole"
$JULIA --project="$REPO_ROOT/scripts" --threads="$NPROC" "$REPO_ROOT/scripts/main.jl" jipole.toml > output/jipole.log 2>&1 || {
    tail -20 output/jipole.log >&2
    log "Jipole failed"
    exit 1
}

"$PYTHON" check.py output/jipole.h5 output/ipole.h5 output/comparison.png
