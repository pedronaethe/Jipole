#!/usr/bin/env bash
# Gradients of the polarized image: automatic differentiation against finite differences.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ../common.sh

#ensure we have python with necessary libraries and julia
#these functions are defined in common.sh
ensure_python
ensure_julia
dump="$(ensure_file sample_dump_SANE_a+0.94_MKS_0900.h5 \
    https://dataverse.harvard.edu/api/access/datafile/12137142 dc8cf9b45136cd9b28bdb8ed197cfd69)"

#redirect logs and images to output folder
rm -rf output
mkdir -p output
ln -s "$dump" output/dump.h5

log "Running Jipole to generate the polarized image and all its gradients through AD"
run_jipole grads.toml

# Central finite differences with dP = h_rel * P, the step of test/img_grads.
h_rel=1e-8
log "Running Jipole for the finite differences (two polarized images per parameter)"
$JULIA --project="$REPO_ROOT/scripts" --threads="$NPROC" fd.jl grads.toml output/fd.h5 "$h_rel" > output/fd.log 2>&1 || {
    tail -20 output/fd.log >&2
    log "Jipole failed on the finite differences"
    exit 1
}

"$PYTHON" check.py output/ad.h5 output/fd.h5 output/comparison.png
