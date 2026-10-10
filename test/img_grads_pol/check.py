import os
import sys

import h5py
import numpy as np

sys.dont_write_bytecode = True  # keep test/ free of __pycache__
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pol_common import GREEN, RED, NC, STOKES, CMAP_SIGNED, INK, plt

# Same threshold as the intensity gradient test (test/img_grads/check.py).
NMSE_MAX = 1e-5
# Fraction of pixels left out of the norm, the ones with the largest difference.
#
# A finite difference is only a derivative where the image is a smooth function of the
# parameter. For the parameters that move the rays (MBH, ro, th, phi, sourceD) it is not
# smooth everywhere: a ray next to the critical curve, or one whose last step enters or leaves
# the emitting region, changes by a finite amount for an arbitrarily small change of the
# parameter, and the finite difference there is that jump divided by the step. One such pixel
# can hold most of the squared difference of the whole image, in the unpolarized gradient just
# as in the polarized ones. Automatic differentiation gives the derivative of the smooth part,
# so those few pixels are excluded; the untrimmed norm is printed next to the trimmed one.
TRIM = 0.01


def nmse(a, ref, trim=0.0):
    d = ((a - ref) ** 2).ravel()
    r = (ref ** 2).ravel()
    if trim > 0:
        keep = np.argsort(d)[: len(d) - int(round(trim * len(d)))]
        d, r = d[keep], r[keep]
    return d.sum() / r.sum() if r.sum() > 0 else d.sum()


def main(ad_path, fd_path, png_path):
    passed = True
    with h5py.File(ad_path, "r") as fa, h5py.File(fd_path, "r") as ff:
        # 1. The images of the dual-number run are those of the plain run.
        print("images of the AD run against the plain run:")
        for name, a, b in [("unpol", fa["unpol"][()], ff["unpol"][()])] + \
                [(f"Stokes {STOKES[s]}", fa["pol"][..., s], ff["pol"][..., s]) for s in range(4)] + \
                [("tauF", fa["pol"][..., 4], ff["pol"][..., 4])]:
            err = nmse(a, b)
            ok = np.all(np.isfinite(a)) and err < 1e-20
            passed &= ok
            print(f"  {name:9s} NMSE {err:.1e}   {GREEN + 'PASS' + NC if ok else RED + 'FAIL' + NC}")

        # 2. Gradients: AD against central finite differences.
        params = [k for k in ff["fd"].keys()]
        print(f"\ngradients, AD against finite differences: NMSE over all pixels / without the worst {100 * TRIM:g}% "
              f"(threshold {NMSE_MAX:.0e} on the latter)")
        print(f"{'parameter':10s} " + " ".join(f"{n:>19s}" for n in ["unpol"] + [f"Stokes {s}" for s in STOKES]))
        rows = {}
        for p in params:
            if f"grad/{p}" not in fa or f"grad_pol/{p}" not in fa:
                print(f"{p:10s} missing in {ad_path}")
                passed = False
                continue
            pairs = [(fa[f"grad/{p}"][()], ff[f"fd/{p}"][()])]
            pairs += [(fa[f"grad_pol/{p}"][..., s], ff[f"fd_pol/{p}"][..., s]) for s in range(4)]
            finite = all(np.all(np.isfinite(a)) for a, _ in pairs)
            full = [nmse(a, b) for a, b in pairs]
            trimmed = [nmse(a, b, TRIM) for a, b in pairs]
            ok = finite and all(t < NMSE_MAX for t in trimmed)
            passed &= ok
            rows[p] = pairs
            cells = " ".join(f"{f:9.1e}/{t:9.1e}" for f, t in zip(full, trimmed))
            print(f"{p:10s} {cells}   {GREEN + 'PASS' + NC if ok else RED + 'FAIL' + NC}")

    # Figure: AD gradient of Stokes Q and its difference from the finite difference, per parameter.
    n = len(rows)
    fig, axes = plt.subplots(2, n, figsize=(1.9 * n + 0.6, 4.4), constrained_layout=True, squeeze=False)
    for col, (p, pairs) in enumerate(rows.items()):
        a, b = pairs[2]  # Stokes Q
        vmax = max(np.abs(a).max(), 1e-300)
        axes[0, col].imshow(a, origin="lower", cmap=CMAP_SIGNED, vmin=-vmax, vmax=vmax)
        axes[0, col].set_title(f"∂Q/∂{p}")
        d = a - b
        dmax = max(np.percentile(np.abs(d), 100 * (1 - TRIM)), 1e-300)
        axes[1, col].imshow(d, origin="lower", cmap=CMAP_SIGNED, vmin=-dmax, vmax=dmax)
        axes[1, col].set_title(f"AD − FD\n(scale {dmax / vmax:.0e} of above)", fontsize=8)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        for side in ("top", "right"):
            ax.spines[side].set_visible(True)
    fig.suptitle("Gradient of Stokes Q by automatic differentiation (each panel on its own symmetric scale)", color=INK)
    fig.savefig(png_path, dpi=110)
    plt.close(fig)
    print(f"wrote {png_path}")

    print("PASS" if passed else "FAIL")
    return passed


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("usage: python check.py AD_H5 FD_H5 PNG")
    sys.exit(0 if main(*sys.argv[1:]) else 1)
