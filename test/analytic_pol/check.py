import os
import sys

import h5py
import numpy as np

sys.dont_write_bytecode = True  # keep test/ free of __pycache__
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pol_common import GREEN, RED, NC, INK, MUTED, SERIES, plt

# Jipole against ipole: same norm and threshold as the intensity test (test/img_grmhd/check.py).
NMSE_MAX = 1e-10
# Jipole's solver against the closed-form solution: the agreement quoted for all codes in
# arXiv:2303.12004, section 4.1. The solver splits each step into rotation and
# emission/absorption parts, which is exact when only one of the two is present (the
# emission/absorption test) and second-order accurate otherwise (the rotation test). At ipole's
# default step of 0.01 the splitting error of the rotation test is 1.3e-5, in ipole as well, so
# the threshold is applied at half that step and the order of convergence is checked too.
EXACT_MAX = 1e-5
ORDER_MIN = 1.9

STOKES = "IQUV"
# Table 1 of arXiv:2303.12004.
COEFFS = {"iq": dict(jI=2.0, jQ=1.0, aI=1.0, aQ=1.2), "quv": dict(jQ=0.1, jU=0.1, jV=0.1, rQ=10.0, rV=-4.0)}
# Stokes parameters that do not vanish in each test.
ACTIVE = {"iq": [0, 1], "quv": [1, 2, 3]}
TITLES = {"iq": "Emission/absorption test", "quv": "Rotation test"}


def exact(name, s):
    """Closed-form solutions for zero initial Stokes vector (Dexter 2016, appendix C)."""
    S = np.zeros((4, len(s)))
    c = COEFFS[name]
    if name == "iq":
        jI, jQ, aI, aQ = c["jI"], c["jQ"], c["aI"], c["aQ"]
        a = aI + aQ
        pre = 1 / (a * (aI - aQ))
        f1 = 1 - (np.exp(-a * s) / 2) * (1 + np.exp(2 * aQ * s))
        f2 = (np.exp(-a * s) / 2) * (1 - np.exp(2 * aQ * s))
        S[0] = pre * ((jI * aI - jQ * aQ) * f1 + (jI * aQ - jQ * aI) * f2)
        S[1] = pre * ((jQ * aI - jI * aQ) * f1 + (jQ * aQ - jI * aI) * f2)
    else:
        jQ, jU, jV, rQ, rV = c["jQ"], c["jU"], c["jV"], c["rQ"], c["rV"]
        r = np.hypot(rQ, rV)
        S[1] = (rQ / r**2) * (jQ * rQ + jV * rV) * s - (rV / r**3) * (jV * rQ - jQ * rV) * np.sin(r * s) \
            - (jU * rV / r**2) * (1 - np.cos(r * s))
        S[2] = ((jQ * rV - jV * rQ) / r**2) * (1 - np.cos(r * s)) + (jU / r) * np.sin(r * s)
        S[3] = (rV / r**2) * (jQ * rQ + jV * rV) * s - (rQ / r**3) * (jQ * rV - jV * rQ) * np.sin(r * s) \
            + (jU * rQ / r**2) * (1 - np.cos(r * s))
    return S


def read_ipole(path, nstep):
    """Per-step plasma-frame Stokes parameters of the first pixel, and that pixel of the image.

    ipole's ldi2 model appends the steps of all pixels to the same arrays, without resetting
    its affine-parameter counter, so only the first `nstep` entries belong to pixel (0, 0).
    """
    with h5py.File(path, "r") as f:
        lam = f["lam"][:nstep]
        S = np.array([f[k][:nstep] for k in STOKES])
        pol = f["pol"][0, 0, :]
        nrec = np.count_nonzero(f["lam"][()])
    return lam, S, pol, nrec


def nmse(a, ref):
    return np.sum((a - ref) ** 2) / np.sum(ref ** 2)


def status(ok):
    return f"{GREEN}PASS{NC}" if ok else f"{RED}FAIL{NC}"


def main(jipole_path, ipole_dir, png_path):
    passed = True
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2), constrained_layout=True)

    with h5py.File(jipole_path, "r") as fj:
        for row, name in enumerate(["iq", "quv"]):
            act = ACTIVE[name]

            # 1. Stokes-space solver against the closed form.
            # h5py reads Jipole's (4, nstep) arrays as (nstep, 4).
            lam_s = fj[f"{name}/solver/lam"][()]
            S_s = fj[f"{name}/solver/stokes"][()].T
            S_e = exact(name, lam_s)
            lam_h = fj[f"{name}/solver_half/lam"][()]
            S_h = fj[f"{name}/solver_half/stokes"][()].T
            S_eh = exact(name, lam_h)
            print(f"{TITLES[name]}: solver against the closed form, 0 < lambda <= {lam_s[-1]:g}")
            for s in act:
                err = np.abs(S_s[s] - S_e[s]).max()
                err_h = np.abs(S_h[s] - S_eh[s]).max()
                ok = err_h < EXACT_MAX
                line = f"  Stokes {STOKES[s]}   max |error| {err:.3e} (step 0.01), {err_h:.3e} (step 0.005)"
                if name == "quv":
                    order = np.log2(err / err_h)
                    ok &= order > ORDER_MIN
                    line += f", order {order:.2f}"
                passed &= ok
                print(f"{line}   threshold {EXACT_MAX:.0e}   {status(ok)}")

            # 2. Full pipeline against ipole, step by step and at the camera.
            lam_p = fj[f"{name}/pipeline/lam"][()]
            S_p = fj[f"{name}/pipeline/stokes"][()].T
            pol_p = fj[f"{name}/pipeline/pol"][()]
            lam_i, S_i, pol_i, nrec = read_ipole(os.path.join(ipole_dir, f"ipole_{name}.h5"), len(lam_p))
            print(f"{TITLES[name]}: full pipeline against ipole, {len(lam_p)} steps to lambda = {lam_p[-1]:.6f}")
            ok = nrec == 4 * len(lam_p) and np.allclose(lam_p, lam_i, rtol=1e-9)
            passed &= ok
            print(f"  steps: ipole recorded {nrec} for 4 pixels, Jipole took {len(lam_p)} for one   {status(ok)}")
            scale = np.abs(S_i[act]).max()
            for s in range(4):
                if s in act:
                    err = nmse(S_p[s], S_i[s])
                    ok = err < NMSE_MAX
                    print(f"  Stokes {STOKES[s]}   NMSE {err:.3e}   threshold {NMSE_MAX:.0e}   {status(ok)}")
                else:
                    # Vanishes in the exact solution; both codes only hold round-off there.
                    err = max(np.abs(S_p[s]).max(), np.abs(S_i[s]).max()) / scale
                    ok = err < 1e-8
                    print(f"  Stokes {STOKES[s]}   max |value| / max signal {err:.1e}   threshold 1e-08   {status(ok)}")
                passed &= ok
            err = nmse(pol_p[:4], pol_i[:4])
            ok = err < NMSE_MAX
            passed &= ok
            print(f"  camera pixel (I, Q, U, V): Jipole {pol_p[:4]}, ipole {pol_i[:4]}")
            print(f"  camera pixel NMSE {err:.3e}   threshold {NMSE_MAX:.0e}   {status(ok)}")
            ok = np.isclose(pol_p[4], pol_i[4], rtol=1e-9, atol=1e-12)
            passed &= ok
            print(f"  Faraday depth: Jipole {pol_p[4]:.10g}, ipole {pol_i[4]:.10g}   {status(ok)}")

            # Figure: solution, solver error, and pipeline minus ipole. One colour per Stokes
            # parameter throughout; every line is labelled at its end, and the legend of the
            # first panel of the row holds for the other two.
            def end_label(ax, x, y, s):
                ax.annotate(STOKES[s], (x[-1], y[-1]), xytext=(4, 0), textcoords="offset points",
                            color=INK, va="center")

            ax = axes[row, 0]
            for s in act:
                ax.plot(lam_s, S_s[s], color=SERIES[s], label=f"Stokes {STOKES[s]}")
                ax.plot(lam_s, S_e[s], color=INK, lw=0.7)
                end_label(ax, lam_s, S_s[s], s)
            ax.plot([], [], color=INK, lw=0.7, label="closed form")
            ax.set_title(f"{TITLES[name]}: Jipole solver")
            ax.legend(loc="best")
            ax = axes[row, 1]
            for s in act:
                ax.plot(lam_s, S_s[s] - S_e[s], color=SERIES[s])
                end_label(ax, lam_s, S_s[s] - S_e[s], s)
            ax.set_title("Jipole solver − closed form (step 0.01)")
            ax = axes[row, 2]
            for s in act:
                ax.plot(lam_p, S_p[s] - S_i[s], color=SERIES[s])
                end_label(ax, lam_p, S_p[s] - S_i[s], s)
            ax.set_title("Jipole pipeline − ipole")
            for ax in axes[row]:
                ax.set_xlabel("λ")
                ax.grid(True)
                ax.axhline(0, color="#c3c2b7", lw=0.8, zorder=0)
                ax.margins(x=0.06)

    fig.savefig(png_path, dpi=110)
    plt.close(fig)
    print(f"wrote {png_path}")
    print("PASS" if passed else "FAIL")
    return passed


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("usage: python check.py JIPOLE_H5 IPOLE_OUTPUT_DIR PNG")
    sys.exit(0 if main(*sys.argv[1:]) else 1)
