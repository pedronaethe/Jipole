#!/usr/bin/env bash
# Brisk-light correctness tests.
#
# Unlike the other suites these need a *series* of dumps, not one file, since
# brisk light is only meaningful when there is more than one epoch to interpolate
# between. There is no small public series to download, so this suite skips
# unless one is pointed at:
#
#     JIPOLE_DUMPDIR=/path/to/dumps bash test/run_all.sh
#
# Roughly 10 dumps at a few M spacing is enough; the tests pick their own small
# geometry so the delay support fits.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ../common.sh

if [[ -z "${JIPOLE_DUMPDIR:-}" ]]; then
    log "SKIP: set JIPOLE_DUMPDIR to a directory of GRMHD dumps to run the brisk-light tests"
    exit 0
fi
[[ -d "$JIPOLE_DUMPDIR" ]] || { log "JIPOLE_DUMPDIR is not a directory: $JIPOLE_DUMPDIR"; exit 1; }

ensure_julia

rm -rf output
mkdir -p output

# Each test asserts internally and exits non-zero on failure, so the suite's
# result is just their conjunction. They are ordered cheapest first.
for t in exactness slowlight_limit differentiable; do
    log "Running $t.jl"
    if ! $JULIA --project="$REPO_ROOT/scripts" --threads="$NPROC" \
            "$t.jl" "$JIPOLE_DUMPDIR" > "output/$t.log" 2>&1; then
        tail -30 "output/$t.log" >&2
        log "FAILED: $t"
        exit 1
    fi
    grep -E "PASSED" "output/$t.log" | tail -1 >&2 || true
done

log "all brisk-light tests passed"
