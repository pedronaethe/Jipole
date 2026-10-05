import os
import sys

import numpy as np

sys.dont_write_bytecode = True  # keep test/ free of __pycache__
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pol_common import GREEN, RED, NC, read_ipole, read_jipole, nmse, print_summary, stokes_figure

# Same norm and threshold as the intensity test (test/img_grmhd/check.py).
NMSE_MAX = 1e-10
# The disk emits no circular polarization and nothing along the rays can create it, so
# Stokes V vanishes in both codes; it is checked against the brightest pixel instead.
V_MAX = 1e-10


def main(jipole_path, ipole_path, png_path):
    unpol_j, pol_j, _ = read_jipole(jipole_path)
    unpol_i, pol_i, header_i = read_ipole(ipole_path)

    if unpol_j.shape != unpol_i.shape:
        print(f"image sizes differ: Jipole {unpol_j.shape}, ipole {unpol_i.shape}")
        print("FAIL")
        return False

    print_summary("ipole", pol_i)
    print_summary("Jipole", pol_j)

    images = [("unpol", unpol_j, unpol_i)]
    images += [(f"Stokes {name}", pol_j[..., s], pol_i[..., s]) for s, name in enumerate("IQU")]

    passed = True
    print(f"{'image':10s} {'NMSE':>10s} {'max |diff|':>12s} {'max |ipole|':>12s}   threshold {NMSE_MAX:.0e}")
    for name, a, b in images:
        err = nmse(a, b)
        ok = err < NMSE_MAX
        passed &= ok
        status = f"{GREEN}PASS{NC}" if ok else f"{RED}FAIL{NC}"
        print(f"{name:10s} {err:10.3e} {np.abs(a - b).max():12.3e} {np.abs(b).max():12.3e}   {status}")

    imax = np.abs(pol_i[..., 0]).max()
    v_j, v_i = np.abs(pol_j[..., 3]).max() / imax, np.abs(pol_i[..., 3]).max() / imax
    ok = v_j <= V_MAX and v_i <= V_MAX
    passed &= ok
    status = f"{GREEN}PASS{NC}" if ok else f"{RED}FAIL{NC}"
    print(f"Stokes V   max |V| / max I: Jipole {v_j:.1e}, ipole {v_i:.1e}   threshold {V_MAX:.0e}   {status}")

    dx = header_i.get("camera/dx", None)
    extent = None if dx is None else [-dx / 2, dx / 2, -dx / 2, dx / 2]
    stokes_figure(pol_j, pol_i, png_path, "Polarized thin disk (axes in r_g)", extent=extent)

    print("PASS" if passed else "FAIL")
    return passed


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("usage: python check.py JIPOLE_H5 IPOLE_H5 PNG")
    sys.exit(0 if main(*sys.argv[1:]) else 1)
