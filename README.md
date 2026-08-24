![GitHub last commit](https://img.shields.io/github/last-commit/pedronaethe/Jipole) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/pedronaethe/Jipole/blob/master/License.txt) [![GitHub repo stars](https://img.shields.io/github/stars/pedronaethe/Jipole?style=social)](https://github.com/pedronaethe/Jipole)

# Jipole

A Julia-based radiative transfer code for curved spacetimes with automatic differentiation capabilities. More details available on [the paper published in ApJ](https://iopscience.iop.org/article/10.3847/1538-4357/ae16a0).

## Overview

Jipole is an ipole-based Julia implementation designed to perform radiative transfer calculations in curved spacetimes, with a particular focus on black hole imaging. The code leverages Julia's automatic differentiation (autodiff) to compute derivatives of input parameters, enabling gradient based optimization methods. For this project, we used [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl) ([Revels et al. 2016](https://arxiv.org/abs/1607.07892)).


## Current Development Status

The current version of Jipole is capable of producing images for Iharm3D file types. We also have implemented the test problems described in  **Section 3.2** of [Gold et al. 2020](https://iopscience.iop.org/article/10.3847/1538-4357/ab96c6) and thin disk model as described in the [Prather et al. 2023](http://iopscience.iop.org/article/10.3847/1538-4357/acc586).

The code is currently able to perform slow-light runs and has been compared with the well estabished code [Blacklight](https://github.com/c-white/blacklight) ([C. White. 2022](https://iopscience.iop.org/article/10.3847/1538-4365/ac77ef/meta)) and [ipole](https://github.com/moscibrodzka/ipole) ([Moscibrodzka & Gammie 2017](https://arxiv.org/abs/1712.03057)).


## Installation and Setup

### First-Time Setup

If this is your first time using Jipole, you'll need to set up the Julia environment and install the required dependencies:

1. **Navigate to the project directory**:
   ```bash
   cd /path/to/jipole
   ```

2. **Start Julia with multithreading capabilities**:
   ```bash
   julia --project="."
   ```

3. **Install Packages**
   ```julia
   using Pkg
   Pkg.instantiate()
   ```

This command will install all the packages specified in the `Project.toml` and `Manifest.toml` files.

### Jupyter Kernel Installation

To use Jipole with Jupyter notebooks, install the project-specific kernel:

```julia
using IJulia
IJulia.installkernel("Jipole", "--project=" * Base.current_project())
```

This creates a dedicated Jupyter kernel that automatically loads the Jipole project environment.

## Package Structure

Jipole is a regular Julia package (`src/Jipole.jl`). Every file under `src/` is its own
capitalized submodule, reachable as `Jipole.<ModuleName>`, e.g. `Jipole.Camera.camera_position`,
`Jipole.Geodesics.get_pixel`, `Jipole.Radiation.integrate_emission!`.

Three interchangeable emission models are provided as submodules — `Jipole.Analytic`,
`Jipole.ThinDisk`, and `Jipole.Iharm` — each defining its own parameters type
(`AnalyticParams`, `ThinDiskParams`, `IharmParams`). Which model runs is decided by which
parameters object you construct and pass around, via Julia's multiple dispatch — there is no
global model switch to edit.

```julia
using Jipole

# Analytic torus (Gold et al. 2020)
model = Jipole.Analytic.AnalyticParams(bhspin, Rout, cstartx, cstopx, MBH)

# Thin disk
model = Jipole.ThinDisk.ThinDiskParams(bhspin, Rout, cstartx, cstopx, MBH, Mdot)

# GRMHD simulation dump
model = Jipole.Iharm.read_header(dump_filepath, MBH)
```

From there, the same calls (`Jipole.Camera.camera_position(...)`, `Jipole.Geodesics.get_pixel(...)`,
`Jipole.Radiation.integrate_emission!(...)`) work regardless of which model you constructed —
see `example_notebooks/GenerateImages.ipynb` for the analytic/thin-disk models side by side, and
`example_notebooks/GenerateImageGRMHD.ipynb` for the GRMHD model.

## Running Jipole

### Starting the Environment

1. **Navigate to the project directory**:
   ```bash
   cd /path/to/jipole
   ```

2. **Start Julia with multithreading capabilities**:
   ```bash
   JULIA_NUM_THREADS=xx julia --project="."
   ```
   
   Replace `xx` with the number of CPU cores you want to utilize.

3. **Launch JupyterLab**:
   ```julia
   using IJulia
   IJulia.jupyterlab(dir=pwd())
   ```

### Using Jupyter Notebooks

1. Open your web browser and navigate to the JupyterLab interface (typically `http://localhost:8888`)
2. When creating or opening a notebook, ensure you select the Jipole kernel for this project from the kernel menu
   (install one with `using IJulia; IJulia.installkernel("Jipole", "--project=" * abspath("."))` if you don't have one yet)
3. Every notebook simply starts with `using Jipole`.

### Notebooks Overview

- **ComputeGeodesics.ipynb**  
  Computes geodesics for each pixel and allows for **debugging and visualization** of the trajectories. Useful for inspecting geodesics before integrating intensity.  

- **GenerateImage.ipynb**  
  Computes the **final intensity map** for thin-disk or analytical models, performing **forward integration of emission** along geodesics.  

- **GenerateImageGRMHD.ipynb**  
Computes the **final intensity map** for a GRMHD Iharm3D snapshot, performing **forward integration of emission** along geodesics.  

- **Autodiff.ipynb**  
  Performs **differentiable ray tracing** to compute **derivatives of the image intensity** with respect to parameters like black hole spin (`a`) and observer inclination (`θ`). Uses the **conjugate gradient algorithm** to recover ground truth parameters from a computed intensity map, demonstrating **gradient-based parameter estimation**.


## Script Overview

Two standalone scripts complement the notebooks for command-line use, both under `scripts/`.

### `generate_image.jl`

Raytraces one or more images from a single TOML configuration file. Every parameter Jipole accepts is read from that file, falling back to a documented default for anything left out. It will shout warnings if the parameter is not identified. `scripts/pars/example_par.toml` is an example.

```bash
julia --project="." --threads=xx scripts/generate_image.jl scripts/pars/example_par.toml
```

What gets produced depends on `[dump].dump_filepath`:
- A single file → one output image.
- A directory → one image per dump whose index falls in `[t_init, t_final]` (every dump in the directory, if left unset).

Setting `[physical].slow_light = true` switches to time-dependent rendering instead: every pixel's geodesic is traced once, then radiative transfer is re-integrated as a sliding 3-dump window advances through simulation time, producing a movie (one frame per `[slowlight].image_cadence`). It still reads its dump sequence from `[dump]`, but requires `dump_filepath` to be a directory. Slow-light output currently goes to `../slow_sims/<timestamp>/`, not to `[output].filename`.

### `plot_imgs.jl`

Batch-plots every `.h5` image in a folder (as produced by `generate_image.jl`) into PNG heatmaps, saved to a `figs/` subfolder created alongside them. Files are plotted in parallel across threads.

```bash
julia --project="." --threads=xx scripts/plot_imgs.jl path/to/results_folder
julia --project="." --threads=xx scripts/plot_imgs.jl path/to/results_folder vmin vmax   # fixed color scale across the whole batch
```

## References

- Gold, R. et al. 2020, ApJ, 897, 148: [Verification of Radiative Transfer Schemes for the EHT](https://iopscience.iop.org/article/10.3847/1538-4357/ab96c6)
- Moscibrodzka, M. & Gammie, C. F. 2017, arXiv:1712.03057: [ipole – semi-analytic scheme for relativistic polarized radiative transport](https://academic.oup.com/mnras/article/475/1/43/4712230)
- Revels, J., Lubin, M., and Papamarkou, T. 2016, arXiv:1607.07892: [Forward-Mode Automatic Differentiation in Julia](https://arxiv.org/abs/1607.07892)
- Naethe Motta, P. et al. 2025, ApJ, 995, 56: [Jipole: A Differentiable ipole-based Code for Radiative Transfer in Curved Spacetimes](https://iopscience.iop.org/article/10.3847/1538-4357/ae16a0)
- Naethe Motta, P. et al. 2026, ApJ, 1004, 2: [Sensitivities of Black Hole Images from General Relativistic Magnetohydrodynamic Simulations](https://iopscience.iop.org/article/10.3847/1538-4357/ae733f)


## Contact


Pedro Naethe Motta at pedronaethemotta [at] usp [dot] br






