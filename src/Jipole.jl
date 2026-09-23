"""
Jipole: a Julia-based radiative transfer code for curved spacetimes, with
automatic differentiation capabilities.

Every submodule is reachable as `Jipole.<ModuleName>`, e.g.
`Jipole.Camera.camera_position`. Three interchangeable emission models are
provided — [`Jipole.Analytic`](@ref), [`Jipole.ThinDisk`](@ref), and
[`Jipole.Iharm`](@ref) — each defining its own parameters type
(`<: Jipole.AbstractModels.AbstractModel`); which model runs is decided by
which parameters object is passed in, via multiple dispatch, rather than
by a global model switch.
"""
module Jipole

include("model_base.jl")
include("constants.jl")
include("geo_types.jl")
include("debug_functions.jl")
include("coords.jl")
include("metrics.jl")
include("camera.jl")
include("utils.jl")
include("tetrads.jl")
include("maxwell_juettner.jl")
include("grid.jl")
include("radiation.jl")
include("geodesics.jl")
include("imaging.jl")
include("models/analytic.jl")
include("models/thin_disk.jl")
include("models/iharm.jl")
include("models/kharma.jl")

include("output.jl")
include("utils_gpu.jl")
include("slowlight.jl")


using PrecompileTools: @setup_workload, @compile_workload
using TOML


#check if the dump is there
const _WORKLOAD_DUMP = joinpath(@__DIR__, "..", "test", "data", "tiny_dump.h5")

# Render a tiny image at precompile time, the same way scripts/generate_image.jl does, so the
# compiled code is saved with the package instead of being JIT-compiled on every run.
if isfile(_WORKLOAD_DUMP)
    @setup_workload begin
        tiny_dump = joinpath(@__DIR__, "..", "test", "data", "tiny_dump.h5")
        config_file = tempname() * ".toml"
        output_file = tempname() * ".h5"
        write(config_file, "[camera]\ntheta_o = 163.0\n")
        @compile_workload begin
            redirect_stdio(stdout=devnull, stderr=devnull) do
                config = TOML.parsefile(config_file)
                theta_o = Utils.get_config(config, "camera", "theta_o", 60.0)
                dump_files = Utils.resolve_dump_files(tiny_dump, typemin(Int), typemax(Int))

                model = Iharm.read_header(dump_files[1], 6.2e9; th_beg=1.74e-2, Rlow=1.0, Rhigh=20.0, beta_crit=1.0,
                    sigma_cut=1.0, sigma_cut_high=-1.0, M_unit=Constants.M_UNIT_SANE)
                simulation_data = [Iharm.load_data(dump_files[1], 20.0, model)]

                ro, phi, freq, fov, nx = 1000.0, 0.0, 230e9, 160.0, 4
                SourceD = 16.9e6 * Constants.PC
                DXsize = SourceD / model.L_unit / Constants.MUAS_PER_RAD * fov
                fovx = DXsize / ro
                Rh = 1 + sqrt(1.0 - model.a^2)
                Image = Imaging.raytrace_image(model, simulation_data, ro, theta_o, phi, freq, nx, nx, fovx, fovx, 2000, Rh, 0.0, 0.0)

                scale = Imaging.calculate_scale_factor(DXsize, DXsize, nx, nx, SourceD, model.L_unit)
                Imaging.output_stokes_parameters(Image, freq, scale, nx, nx, SourceD)
                Xcamera = Camera.camera_position(ro, theta_o, phi, model.a, model)
                output_data = Dict{String,Any}("image" => Image, "img_time" => 0.0, "params" => model, "data" => simulation_data[1],
                    "ro" => ro, "theta_o" => theta_o, "phi" => phi, "fovx" => fovx, "fovy" => fovx, "freq" => freq,
                    "SourceD" => SourceD, "scale" => scale, "Xcamera" => Xcamera, "Rhigh" => 20.0)
                Output.generate_output_file(output_file, output_data; format="ipole")
            end
        end
        rm(config_file; force=true)
        rm(output_file; force=true)
    end
end

end
