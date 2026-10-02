import sys

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

NMSE_MAX = 1e-10
HEADER_KEYS = ["camera/nx", "camera/ny", "camera/thetacam", "camera/phicam", "camera/rcam",
               "camera/fovx_dsource", "units/M_unit", "electrons/rhigh", "electrons/rlow",
               "electrons/beta_crit", "sigma_cut", "freqcgs", "dsource"]


def read(path, transpose):
    with h5py.File(path, "r") as f:
        image = f["unpol"][()] * f["header/scale"][()]
        header = {k: f["header/" + k][()] for k in HEADER_KEYS}
        flux = f["Ftot_unpol"][()]
    return (image.T if transpose else image), flux, header


def main(jipole_path, ipole_path, png_path):
    img_j, flux_j, header_j = read(jipole_path, transpose=False)
    img_i, flux_i, header_i = read(ipole_path, transpose=True)

    mismatched = [k for k in HEADER_KEYS if not np.isclose(header_j[k], header_i[k], rtol=1e-6)]
    if mismatched:
        for k in mismatched:
            print(f"setting differs: {k}: Jipole {header_j[k]}, ipole {header_i[k]}")
        print("FAIL")
        return False

    nmse = np.sum((img_j - img_i) ** 2) / np.sum(img_i ** 2)
    rel = np.abs(img_j - img_i) / np.where(img_i > 0, img_i, np.nan)
    print(f"Jipole flux  {flux_j:.12e} Jy")
    print(f"ipole flux   {flux_i:.12e} Jy   relative difference {abs(flux_j / flux_i - 1):.3e}")
    print(f"NMSE         {nmse:.3e}   threshold {NMSE_MAX:.0e}")
    print(f"max relative pixel difference {np.nanmax(rel):.3e}")

    fov = header_i["camera/fovx_dsource"]
    extent = [-fov / 2, fov / 2, -fov / 2, fov / 2]
    vmax = max(img_j.max(), img_i.max())
    vmin = vmax * 1e-8
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, img, title in ((axes[0], img_j, f"Jipole, F = {flux_j:.6f} Jy"),
                           (axes[1], img_i, f"ipole, F = {flux_i:.6f} Jy")):
        im = ax.imshow(np.clip(img, vmin, None), origin="lower", cmap="afmhot",
                       norm=LogNorm(vmin=vmin, vmax=vmax), extent=extent)
        ax.set(xlabel=r"x [$\mu$as]", ylabel=r"y [$\mu$as]", title=title)
        fig.colorbar(im, ax=ax, label="Jy / pixel")
    im = axes[2].imshow(np.clip(rel, 1e-8, 1), origin="lower", cmap="viridis",
                        norm=LogNorm(vmin=1e-8, vmax=1), extent=extent)
    axes[2].set(xlabel=r"x [$\mu$as]", ylabel=r"y [$\mu$as]",
                title=f"|Jipole − ipole| / ipole,  NMSE = {nmse:.2e}")
    fig.colorbar(im, ax=axes[2], label="relative difference")
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    print(f"wrote {png_path}")

    passed = nmse < NMSE_MAX
    print("PASS" if passed else "FAIL")
    return passed


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("usage: python check.py JIPOLE_H5 IPOLE_H5 PNG")
    sys.exit(0 if main(*sys.argv[1:]) else 1)
