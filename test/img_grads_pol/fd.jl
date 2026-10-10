# Finite-difference gradients of the polarized image, to check the automatic-differentiation
# ones written by scripts/main.jl for the same parameter file.
#
# Everything here goes through the ordinary Float64 code path (read_header, load_data,
# raytrace_image_pol), so it shares nothing with the dual-number setup being checked. All
# runs happen in this one Julia session.
#
# Usage: julia --project=scripts --threads=N fd.jl grads.toml OUTPUT_H5 H_REL
using Jipole
using TOML

const HDF5 = Jipole.Output.HDF5
const C = Jipole.Constants

config = TOML.parsefile(ARGS[1])
output_file = ARGS[2]
h_rel = parse(Float64, ARGS[3])

dump = config["dump"]["dump_filepath"]
wrt = Symbol.(config["gradient"]["wrt"])
nx, ny = config["image"]["pixels_x"], config["image"]["pixels_y"]
fov_size = config["image"]["fov_size"]
xoff, yoff = config["image"]["xoff"], config["image"]["yoff"]
freq = config["observing"]["freq"]
maxnstep = config["raytracing"]["maxnstep"]
plasma = config["plasma"]

# The differentiable parameters, by the names used in [gradient].wrt. sourceD is in cm.
p0 = (M_unit=float(config["physical"]["m_unit"]), MBH=config["physical"]["MBH"],
      Rhigh=plasma["Rhigh"], Rlow=plasma["Rlow"], beta_crit=plasma["beta_crit"],
      ro=config["camera"]["ro"], th=config["camera"]["theta_o"], phi=config["camera"]["phi"],
      sourceD=config["observing"]["source_distance_pc"] * C.PC)

"""Unpolarized and polarized image (CGS intensity) for the parameter set `p`."""
function render(p)
    redirect_stdout(devnull) do
        model = Jipole.Iharm.read_header(dump, p.MBH; th_beg=plasma["th_beg"], Rlow=p.Rlow, Rhigh=p.Rhigh,
            beta_crit=p.beta_crit, sigma_cut=plasma["sigma_cut"], sigma_cut_high=plasma["sigma_cut_high"], M_unit=p.M_unit)
        data = [Jipole.Iharm.load_data(dump, p.Rhigh, model)]
        Rh = 1 + sqrt(1.0 - model.a^2)
        DXsize = p.sourceD / model.L_unit / C.MUAS_PER_RAD * fov_size
        fov = DXsize / p.ro
        Jipole.ImagingPol.raytrace_image_pol(model, data, p.ro, p.th, p.phi, freq, nx, ny, fov, fov, maxnstep, Rh, xoff, yoff)
    end
end

mkpath(dirname(output_file))
HDF5.h5open(output_file, "w") do f
    unpol, pol = render(p0)
    write(f, "unpol", unpol)
    write(f, "pol", pol)
    for name in wrt
        value = getproperty(p0, name)
        step = value == 0 ? h_rel : h_rel * abs(value)
        up_unpol, up_pol = render(merge(p0, NamedTuple{(name,)}((value + step,))))
        dn_unpol, dn_pol = render(merge(p0, NamedTuple{(name,)}((value - step,))))
        write(f, "fd/$name", (up_unpol .- dn_unpol) ./ (2step))
        write(f, "fd_pol/$name", (up_pol .- dn_pol) ./ (2step))
        println("finite difference in $name: step $step")
    end
end
println("Wrote $output_file")
