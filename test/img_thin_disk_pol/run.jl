# Polarized thin-disk image with Jipole, for comparison against ipole's thin_disk model.
# Usage: julia --project=scripts --threads=N run.jl jipole.toml
using Jipole
using StaticArrays
using TOML

config = TOML.parsefile(ARGS[1])
const C = Jipole.Constants

bhspin = config["disk"]["a"]
MBH = config["disk"]["MBH"]
Rout = config["disk"]["Rout"]
ro, th, phi = config["camera"]["ro"], config["camera"]["theta_o"], config["camera"]["phi"]
nx, ny = config["image"]["pixels_x"], config["image"]["pixels_y"]
DX, DY = config["image"]["dx"], config["image"]["dy"]
freq = config["observing"]["freq"]
SourceD = config["observing"]["source_distance_pc"] * C.PC
maxnstep = config["raytracing"]["maxnstep"]

# Same setup as ipole's model/thin_disk/model.c: the grid runs from the horizon to Rout, where
# the geodesics also stop, and the accretion rate is given in units of the Eddington rate.
Rh = 1 + sqrt(1.0 - bhspin^2)
cstartx = MVector{4,Float64}(0.0, log(Rh), 0.0, 0.0)
cstopx = MVector{4,Float64}(0.0, log(Rout), 1.0, 2.0 * π)
Mdotedd = 4 * π * C.GNEWT * MBH * C.MSUN * C.MP / 0.1 / C.CL / C.SIGMA_THOMPSON
Mdot = config["disk"]["Mdot"] * Mdotedd

model = Jipole.ThinDisk.ThinDiskParams(bhspin, Rout, cstartx, cstopx, MBH, Mdot, Rout)

fovx = DX / ro
fovy = DY / ro
unpol, pol = Jipole.ImagingPol.raytrace_image_pol(model, nothing, ro, th, phi, freq, nx, ny, fovx, fovy, maxnstep, Rh, 0.0, 0.0)

scale = Jipole.Imaging.calculate_scale_factor(DX, DY, nx, ny, SourceD, model.L_unit)
Jipole.Imaging.output_stokes_parameters(unpol, freq, scale, nx, ny, SourceD)
Jipole.ImagingPol.output_stokes_parameters_pol(pol, freq, scale, SourceD)

output_file = config["output"]["filename"]
mkpath(dirname(output_file))
Jipole.Output.generate_output_file(output_file, Dict{String,Any}(
        "unpol" => unpol, "pol" => pol, "scale" => scale, "freqcgs" => freq, "dsource" => SourceD,
        "dx" => DX, "dy" => DY, "a" => bhspin, "r_isco" => model.r_isco, "T0" => model.T0);
    format="generic")
println("Wrote $output_file")
