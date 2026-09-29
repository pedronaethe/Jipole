import os
import sys

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, SymLogNorm
import numpy as np

NMSE_MAX = 1e-5
GREEN, RED, NC = "\033[0;32m", "\033[0;31m", "\033[0m"
CM_PER_PC = 3.085678e18
STEP_TO_AD_UNITS = {"sourceD": CM_PER_PC}


def read_steps(fd_dir):
    steps = {}
    with open(os.path.join(fd_dir, "steps.txt")) as f:
        for line in f:
            if line.strip():
                name, step = line.split()
                steps[name] = float(step) * STEP_TO_AD_UNITS.get(name, 1.0)
    return steps


def read_unpol(path):
    with h5py.File(path, "r") as f:
        return f["unpol"][()], f["header/scale"][()]


def nmse(ad, fd):
    denom = np.sum(fd ** 2)
    if denom == 0:
        denom = np.sum(ad ** 2)
    return 0.0 if denom == 0 else np.sum((ad - fd) ** 2) / denom


def main(ad_path, fd_dir, png_path):
    steps = read_steps(fd_dir)
    results = []
    with h5py.File(ad_path, "r") as f:
        scale = f["header/scale"][()]
        fov = f["header/camera/fovx_dsource"][()]
        for name, step in steps.items():
            key = f"grad/{name}"
            if key not in f:
                print(f"{name:10s} missing {key} in {ad_path}")
                results.append((name, None, None, np.inf))
                continue
            ad = f[key][()]
            plus, _ = read_unpol(os.path.join(fd_dir, f"{name}_plus.h5"))
            minus, _ = read_unpol(os.path.join(fd_dir, f"{name}_minus.h5"))
            fd = (plus - minus) / (2 * step)
            results.append((name, ad, fd, nmse(ad, fd)))

    print(f"{'parameter':10s} {'dF/dP AD':>16s} {'dF/dP FD':>16s} {'NMSE':>10s}")
    for name, ad, fd, err in results:
        if ad is None:
            continue
        status = f"{GREEN}PASS{NC}" if err < NMSE_MAX else f"{RED}FAIL{NC}"
        print(f"{name:10s} {ad.sum() * scale:16.8e} {fd.sum() * scale:16.8e} {err:10.2e}  {status}")

    plotted = [r for r in results if r[1] is not None]
    if plotted:
        extent = [-fov / 2, fov / 2, -fov / 2, fov / 2]
        fig, axes = plt.subplots(len(plotted), 3, figsize=(17, 5 * len(plotted)), squeeze=False)
        for row, (name, ad, fd, err) in zip(axes, plotted):
            vmax = max(np.abs(ad).max(), np.abs(fd).max(), np.finfo(float).tiny)
            norm = SymLogNorm(linthresh=vmax * 1e-8, vmin=-vmax, vmax=vmax)
            for ax, img, title in ((row[0], ad, f"AD  dI/d{name}"), (row[1], fd, f"FD  dI/d{name}")):
                im = ax.imshow(img, origin="lower", cmap="RdBu_r", norm=norm, extent=extent)
                ax.set(xlabel=r"x [$\mu$as]", ylabel=r"y [$\mu$as]", title=title)
                fig.colorbar(im, ax=ax, label=f"dI/d{name}")
            rel = np.abs(ad - fd) / np.where(fd != 0, np.abs(fd), np.nan)
            im = row[2].imshow(np.clip(rel, 1e-8, 1), origin="lower", cmap="viridis",
                               norm=LogNorm(vmin=1e-8, vmax=1), extent=extent)
            row[2].set(xlabel=r"x [$\mu$as]", ylabel=r"y [$\mu$as]",
                       title=f"|AD − FD| / |FD|,  NMSE = {err:.2e}")
            fig.colorbar(im, ax=row[2], label="relative difference")
        fig.tight_layout()
        fig.savefig(png_path, dpi=100)
        print(f"wrote {png_path}")

    passed = len(results) > 0 and all(err < NMSE_MAX for _, _, _, err in results)
    print(f"{GREEN}PASS{NC}" if passed else f"{RED}FAIL{NC}")
    return passed


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit("usage: python check.py AD_H5 FD_DIR PNG")
    sys.exit(0 if main(*sys.argv[1:]) else 1)
