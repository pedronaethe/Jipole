#!/usr/bin/env bash
# Constant-coefficient polarized transfer (arXiv:2303.12004, section 3.1): Jipole against the
# closed-form solutions, and against ipole's ldi2 model step by step.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ../common.sh

# ipole's ldi2 model does not link at the pinned commit: it lacks two functions that the
# rest of ipole now calls. Add them (they return zero, as in ipole's other analytic models)
# before building. ensure_ipole iharm makes sure the ipole clone exists first.
ensure_ipole iharm
if ! git -C "$IPOLE_DIR" apply --reverse --check "$PWD/ipole_ldi2_stubs.patch" 2> /dev/null; then
    git -C "$IPOLE_DIR" apply "$PWD/ipole_ldi2_stubs.patch" || {
        log "Could not add the missing stubs to ipole's ldi2 model"
        exit 1
    }
    rm -f "$IPOLE_DIR/ipole_ldi2"
fi
ensure_ipole ldi2
ensure_python
ensure_julia

#redirect logs and results to output folder
rm -rf output
mkdir -p output

# The two parameter files ship with ipole but leave out settings it needs:
#  - emission_type 10 hands the constant coefficients of the model to the solver;
#  - freqcgs must be set (its value is irrelevant here);
#  - a tiny field of view keeps the ray radial;
#  - nx = ny = 1, as in the files, divides by zero in ipole's image loop, hence 2x2.
# One thread, because the model records its steps in a single shared array.
for name in iq quv; do
    log "Running ipole (ldi2_$name)"
    OMP_NUM_THREADS=1 "$IPOLE_BIN" -par "$IPOLE_DIR/model/ldi2/ldi2_$name.par" \
        --emission_type=10 --freqcgs=1 --dx=1e-10 --dy=1e-10 --nx=2 --ny=2 \
        --outfile="output/ipole_$name.h5" > "output/ipole_$name.log" 2>&1 || {
        tail -20 "output/ipole_$name.log" >&2
        log "ipole failed"
        exit 1
    }
done

log "Running Jipole"
$JULIA --project="$REPO_ROOT/scripts" run.jl output/jipole.h5 > output/jipole.log 2>&1 || {
    tail -20 output/jipole.log >&2
    log "Jipole failed"
    exit 1
}

"$PYTHON" check.py output/jipole.h5 output output/comparison.png
