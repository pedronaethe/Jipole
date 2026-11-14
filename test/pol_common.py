"""Helpers shared by the polarization tests' check.py scripts.

Array layout. ipole writes "unpol" as (nx, ny) and "pol" as (nx, ny, 5). Jipole
writes "unpol" as the Julia array (nx, ny), which h5py reads as (ny, nx), and
"pol" as the Julia array (5, nx, ny), which h5py reads as (ny, nx, 5). As in
test/img_grmhd/check.py, ipole's arrays are transposed to Jipole's layout, so
every image here is indexed [y, x] and can be shown with origin="lower".
"""
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

GREEN, RED, NC = "\033[0;32m", "\033[0;31m", "\033[0m"
STOKES = ["I", "Q", "U", "V"]

# Figure style: recessive axes, ink for text, and one colour scale per job.
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
# Magnitude (Stokes I): a heat scale, dark = no emission, as is usual for these images.
CMAP_I = "afmhot"
# Signed quantities (Q, U, V, differences): two opposite hues around a neutral midpoint.
CMAP_SIGNED = LinearSegmentedColormap.from_list(
    "blue_gray_red", ["#0d366b", "#256abf", "#86b6ef", "#f0efec", "#f0a5a4", "#d03b3b", "#7a1414"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": MUTED, "axes.edgecolor": "#c3c2b7",
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK,
    "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "lines.linewidth": 2.0, "font.size": 9, "axes.titlesize": 10, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False,
})


def _scale(f):
    return f["header/scale"][()] if "header" in f else f["scale"][()]


def read_ipole(path):
    """unpol [Jy/px], pol I,Q,U,V [Jy/px] + Faraday depth, in Jipole's [y, x] layout."""
    with h5py.File(path, "r") as f:
        scale = _scale(f)
        unpol = f["unpol"][()].T * scale
        pol = f["pol"][()].transpose(1, 0, 2).copy()
        header = {k: f["header/" + k][()] for k in HEADER_KEYS + EXTRA_KEYS if "header/" + k in f}
    pol[..., :4] *= scale
    return unpol, pol, header


def read_jipole(path):
    with h5py.File(path, "r") as f:
        scale = _scale(f)
        unpol = f["unpol"][()] * scale
        pol = f["pol"][()].copy()
        header = {k: f["header/" + k][()] for k in HEADER_KEYS + EXTRA_KEYS if "header/" + k in f}
    pol[..., :4] *= scale
    return unpol, pol, header


HEADER_KEYS = ["camera/nx", "camera/ny", "camera/thetacam", "camera/phicam", "camera/rcam",
               "camera/fovx_dsource", "units/M_unit", "electrons/rhigh", "electrons/rlow",
               "electrons/beta_crit", "sigma_cut", "freqcgs", "dsource"]
# Read when present, but not part of the settings comparison.
EXTRA_KEYS = ["camera/dx", "camera/dy"]


def headers_match(header_j, header_i, keys=HEADER_KEYS):
    """True if the run settings recorded by the two codes agree (as in test/img_grmhd)."""
    bad = [k for k in keys if k in header_j and k in header_i
           and not np.isclose(header_j[k], header_i[k], rtol=1e-6)]
    for k in bad:
        print(f"setting differs: {k}: Jipole {header_j[k]}, ipole {header_i[k]}")
    return not bad


def nmse(a, ref):
    """Normalized mean squared error, sum((a - ref)^2) / sum(ref^2): the norm of test/img_grmhd."""
    denom = np.sum(ref ** 2)
    return np.sum((a - ref) ** 2) / denom if denom > 0 else np.sum((a - ref) ** 2)


def summary(pol):
    """Image-integrated flux [Jy], net LP and CP [%], and EVPA [deg], as ipole prints them."""
    I, Q, U, V = (pol[..., s].sum() for s in range(4))
    return {"flux": I, "LP": 100 * np.hypot(Q, U) / I, "CP": 100 * V / I,
            "EVPA": np.degrees(0.5 * np.arctan2(U, Q))}


def print_summary(name, pol):
    s = summary(pol)
    print(f"{name:8s} flux {s['flux']:.8e} Jy   LP {s['LP']:.6f} %   CP {s['CP']:.6f} %   EVPA {s['EVPA']:.4f} deg")


def stokes_figure(pol_j, pol_i, png_path, title, extent=None, unit="Jy/px", zoom=None):
    """Rows: ipole, Jipole, Jipole - ipole. Columns: Stokes I, Q, U, V.

    `extent` gives the axis range [x0, x1, y0, y1]; `zoom` limits the view to +-zoom around
    the centre, in the same units.
    """
    fig, axes = plt.subplots(3, 4, figsize=(13, 9.8), constrained_layout=True)
    for s, name in enumerate(STOKES):
        a, b = pol_j[..., s], pol_i[..., s]
        vmax = max(np.abs(b).max(), 1e-300)
        kw = dict(cmap=CMAP_I, vmin=0, vmax=vmax) if s == 0 else dict(cmap=CMAP_SIGNED, vmin=-vmax, vmax=vmax)
        for row, (img, label) in enumerate([(b, "ipole"), (a, "Jipole")]):
            im = axes[row, s].imshow(img, origin="lower", extent=extent, **kw)
            axes[row, s].set_title(f"Stokes {name}, {label}")
            fig.colorbar(im, ax=axes[row, s], shrink=0.85, label=unit)
        d = a - b
        dmax = max(np.abs(d).max(), 1e-300)
        im = axes[2, s].imshow(d, origin="lower", extent=extent, cmap=CMAP_SIGNED, vmin=-dmax, vmax=dmax)
        axes[2, s].set_title(f"Stokes {name}, Jipole − ipole\nNMSE {nmse(a, b):.1e}")
        fig.colorbar(im, ax=axes[2, s], shrink=0.85, label=unit)
    for ax in axes.ravel():
        if extent is None:
            ax.set_xticks([])
            ax.set_yticks([])
        elif zoom is not None:
            ax.set_xlim(-zoom, zoom)
            ax.set_ylim(-zoom, zoom)
        for side in ("top", "right"):
            ax.spines[side].set_visible(True)
    fig.suptitle(title, color=INK)
    fig.savefig(png_path, dpi=110)
    plt.close(fig)
    print(f"wrote {png_path}")
