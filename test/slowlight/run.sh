#!/usr/bin/env bash
# Slow-light regression tests: the time interpolation never extrapolates (tinterp_clamp.jl), and the replay of the
# M87 crash window (crash_replay.jl). They need GRMHD dumps, so the suite skips unless pointed at them:
#
#     JIPOLE_SLOW_DUMPDIR=/path/to/dumps bash test/run_all.sh
#
# tinterp_clamp.jl only reads a grid header from that directory; crash_replay.jl needs the MAD run's
# torus.out0.01715..01717.phdf (on Delta: /work/hdd/bfyx/pnaethemotta/lightmodes/dumps/MAD_10k_14k) and about
# 40 GB of memory, and is skipped when they are not there.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ../common.sh

if [[ -z "${JIPOLE_SLOW_DUMPDIR:-}" ]]; then
    log "SKIP: set JIPOLE_SLOW_DUMPDIR to a directory of GRMHD dumps to run the slow-light tests"
    exit 0
fi
[[ -d "$JIPOLE_SLOW_DUMPDIR" ]] || { log "JIPOLE_SLOW_DUMPDIR is not a directory: $JIPOLE_SLOW_DUMPDIR"; exit 1; }

ensure_julia

rm -rf output
mkdir -p output

tests=(tinterp_clamp)
[[ -f "$JIPOLE_SLOW_DUMPDIR/torus.out0.01717.phdf" ]] && tests+=(crash_replay) ||
    log "SKIP crash_replay: $JIPOLE_SLOW_DUMPDIR has no torus.out0.01715..01717.phdf"
for t in "${tests[@]}"; do
    log "Running $t.jl"
    if ! $JULIA --project="$REPO_ROOT/scripts" --threads="$NPROC" \
            "$t.jl" "$JIPOLE_SLOW_DUMPDIR" > "output/$t.log" 2>&1; then
        tail -30 "output/$t.log" >&2
        log "FAILED: $t"
        exit 1
    fi
    grep -E "PASSED" "output/$t.log" | tail -1 >&2 || true
done

log "all slow-light tests passed"
