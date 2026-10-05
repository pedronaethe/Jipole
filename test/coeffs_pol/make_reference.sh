#!/usr/bin/env bash
# Regenerate reference.dat from ipole's own fit routines. Not run by the test suite: the
# reference only changes if ipole's fits or the input grid below change.
#
# Needs the ipole clone of the test suite (test/ipole, made by any test that calls
# ensure_ipole), a C compiler and GSL. Point CC, CFLAGS and LDFLAGS at GSL if it is not in the
# default search path, e.g. CFLAGS=-I/opt/anaconda3/include LDFLAGS=-L/opt/anaconda3/lib.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
IPOLE_DIR="../ipole"
[[ -d "$IPOLE_DIR/src/symphony" ]] || { echo "no ipole clone at $IPOLE_DIR; run a test that builds ipole first" >&2; exit 1; }

mkdir -p output
${CC:-cc} -std=gnu99 -O2 -DDEBUG=0 ${CFLAGS:-} \
    -I"$IPOLE_DIR/src" -I"$IPOLE_DIR/src/symphony" -I"$IPOLE_DIR/model/iharm" \
    ipole_fits.c "$IPOLE_DIR"/src/symphony/*.c "$IPOLE_DIR/src/radiation.c" \
    ${LDFLAGS:-} -lgsl -lgslcblas -lm -o output/ipole_fits

# Input grid: densities, frequencies, temperatures (down to the 1e-3 floor of the GRMHD
# model, across the underflow of the Bessel functions near 1.35e-3), field strengths, and
# field-wavevector angles from almost aligned to almost anti-aligned; then the exactly
# aligned cases, as get_bk_angle returns them after clamping.
python3 - > output/inputs.dat <<'PY'
from math import pi
for nu in (8.6e10, 2.3e11, 1.0e12):
    for thetae in (1.0e-3, 1.36e-3, 1.4e-3, 2.0e-3, 1.0e-2, 0.1, 1.0, 10.0, 100.0, 1.0e3):
        for b in (1.0e-3, 5.0, 300.0):
            for theta in (1.0e-9, 1.0e-3, 0.3, pi / 2, 2.0, pi - 1.0e-3, pi - 1.0e-9):
                print(f"{1.0e5:.17g} {nu:.17g} {thetae:.17g} {b:.17g} {theta:.17g}")
for thetae in (1.0e-3, 0.02, 0.05, 1.0, 30.0):
    for b in (0.5, 30.0):
        for theta in (0.0, pi):
            print(f"{1.0e5:.17g} {2.3e11:.17g} {thetae:.17g} {b:.17g} {theta:.17g}")
PY

{
    echo "# Ne nu Thetae B theta | jI jQ jV rhoQ rhoV rhoV_dexter Bnu_inv   (ipole $(git -C "$IPOLE_DIR" rev-parse --short HEAD), see ipole_fits.c)"
    paste -d' ' output/inputs.dat <(output/ipole_fits < output/inputs.dat)
} > reference.dat
echo "wrote reference.dat ($(grep -vc '^#' reference.dat) points)"
