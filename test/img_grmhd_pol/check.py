import os
import sys

import numpy as np

sys.dont_write_bytecode = True  # keep test/ free of __pycache__
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pol_common import (GREEN, RED, NC, STOKES, read_ipole, read_jipole, headers_match, nmse,
                        print_summary, stokes_figure)

# Same norm and threshold as the intensity test (test/img_grmhd/check.py).
NMSE_MAX = 1e-10


def main(jipole_path, ipole_path, png_path):
    unpol_j, pol_j, header_j = read_jipole(jipole_path)
    unpol_i, pol_i, header_i = read_ipole(ipole_path)

    if not headers_match(header_j, header_i):
        print("FAIL")
        return False

    print_summary("ipole", pol_i)
    print_summary("Jipole", pol_j)

    # Unpolarized image, the four Stokes parameters, and the Faraday depth.
    images = [("unpol", unpol_j, unpol_i)]
    images += [(f"Stokes {name}", pol_j[..., s], pol_i[..., s]) for s, name in enumerate(STOKES)]
    images += [("tauF", pol_j[..., 4], pol_i[..., 4])]

    passed = True
    print(f"{'image':10s} {'NMSE':>10s} {'max |diff|':>12s} {'max |ipole|':>12s}   threshold {NMSE_MAX:.0e}")
    for name, a, b in images:
        err = nmse(a, b)
        ok = err < NMSE_MAX
        passed &= ok
        status = f"{GREEN}PASS{NC}" if ok else f"{RED}FAIL{NC}"
        print(f"{name:10s} {err:10.3e} {np.abs(a - b).max():12.3e} {np.abs(b).max():12.3e}   {status}")

    fov = header_i.get("camera/fovx_dsource", None)
    extent = None if fov is None else [-fov / 2, fov / 2, -fov / 2, fov / 2]
    stokes_figure(pol_j, pol_i, png_path, "Polarized GRMHD snapshot (axes in μas)", extent=extent, zoom=40)

    print("PASS" if passed else "FAIL")
    return passed


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("usage: python check.py JIPOLE_H5 IPOLE_H5 PNG")
    sys.exit(0 if main(*sys.argv[1:]) else 1)
