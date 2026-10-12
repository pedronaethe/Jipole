"""
Slow-light (time-dependent) rendering: interpolating radiative transfer
between successive GRMHD dumps as the photon travel time becomes
comparable to the simulation's evolution timescale.
"""
module Slowlight

using HDF5
using Printf
using DelimitedFiles
using ProgressMeter
using StaticArrays
using ..Constants
using ..Radiation
using ..Iharm
using ..Kharma
using ..Output
using ..Utils
using ..GeoTypes
using ..Utils_GPU
import ..Polarization
using Dates
export OfSlowLight, update_dump_path, get_specific_dump_time, update_data!,
    process_slowlight_images!


"""
    GPU functions defined in JipoleCUDAExt.jl
"""
function gpu_tile_plan end
function render_round_gpu! end

"""
Slow-light run state: which dump is currently loaded/being advanced to,
and the time window `[tA, tB]` currently bracketed by `simulation_data`.
"""
mutable struct OfSlowLight
    nloaded::Int64
    dump_max::Int64
    ImageCadence::Float64
    tA::Float64
    tB::Float64
    tf::Float64
    current_dumps_path::String
end

"""
    update_dump_path(params_slowlight, all_dumps_path)

Compute the path of the next dump to load, and advance
`params_slowlight.nloaded`.

# Arguments
- `params_slowlight`: Slow-light run state.
- `all_dumps_path`: `Printf`-style format string for the dump sequence
  (e.g. `".../tmp.%05d.h5"`).

# Returns
- The path of the next dump file.
"""
function update_dump_path(params_slowlight::OfSlowLight, all_dumps_path::String)
    dump_idx = params_slowlight.nloaded
    params_slowlight.nloaded += 1

    return Printf.format(Printf.Format(all_dumps_path), dump_idx)
end

"""
    get_specific_dump_time(dump_idx, all_dumps_path)

Read the simulation time recorded in dump `dump_idx`, without loading its
fluid primitives.

# Arguments
- `dump_idx`: Index of the dump in the sequence.
- `all_dumps_path`: `Printf`-style format string for the dump sequence.

# Returns
- The simulation time of the dump.
"""
function get_specific_dump_time(dump_idx::Int64, all_dumps_path::String)
    dump_path = Printf.format(Printf.Format(all_dumps_path), dump_idx)
    # Delegate so that both dump layouts work: `Utils.dump_time` reads the
    # Parthenon `Info/Time` attribute for native KHARMA `.phdf` files and the
    # root `t` dataset for iharm3d files. Reading `t` directly, as this used to,
    # made slow light silently unusable on a KHARMA directory.
    return Utils.dump_time(dump_path)
end

"""
    update_data!(params_slowlight, simulation_data, Rhigh, model, all_dumps_path)

Slide the 3-snapshot window `simulation_data` forward by one dump, loading the
next dump into slot 3. The reader is chosen per file, so both KHARMA and
iharm3d dumps work.

# Arguments
- `params_slowlight`: Slow-light run state, updated with the new time
  window `[tA, tB]`.
- `simulation_data`: 3-element window of loaded GRMHD snapshots.
- `Rhigh`: Electron/ion temperature ratio at high magnetization.
- `model`: Iharm model parameters.
- `all_dumps_path`: `Printf`-style format string for the dump sequence.
"""
function update_data!(params_slowlight::OfSlowLight, simulation_data, Rhigh::Float64, model::Iharm.IharmParams, all_dumps_path::String)
    # Slide the window. The previous version also assigned the evicted snapshot
    # into slot 3 first, which the `load_data` below overwrote immediately -- a
    # dead store, so despite the old docstring no buffer was ever reused.
    simulation_data[1] = simulation_data[2]
    simulation_data[2] = simulation_data[3]

    # Dispatch on the dump itself so a KHARMA directory loads through `Kharma`
    # and an iharm3d one through `Iharm`, matching the fast-light branch. Both
    # readers return an `Iharm.IharmData`, so the window's element type is
    # unchanged.
    path = params_slowlight.current_dumps_path
    reader = Kharma.is_kharma_dump(path) ? Kharma : Iharm
    simulation_data[3] = reader.load_data(path, Rhigh, model;
        advance_path! = () -> (params_slowlight.current_dumps_path = update_dump_path(params_slowlight, all_dumps_path)))

    params_slowlight.tA = simulation_data[1].t
    params_slowlight.tB = simulation_data[2].t

    @info "Loaded data" dump_path = params_slowlight.current_dumps_path tA = params_slowlight.tA tB = params_slowlight.tB
end

# ---------------------------------------------------------------------------
# Polarized state of an open frame
# ---------------------------------------------------------------------------
#
# A polarized slow-light frame carries, per pixel, the coherency tensor N (a
# Hermitian 4x4 complex matrix) and the Faraday depth from one round to the
# next. They are stored packed as POL_STATE_LEN Float64s, `movie_pol[:, i, j, k]`:
# the four (real) diagonal entries, then the real and imaginary parts of the six
# entries above the diagonal, then tauF. 136 bytes per pixel and frame instead of
# 264 for the full complex matrix; N is Hermitian, so nothing is lost.

const POL_STATE_LEN = 17
const POL_UPPER = ((1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4))

@inline function load_pol_state(movie_pol, i, j, k)
    @inbounds begin
        v(n) = movie_pol[n, i, j, k]
        c(m) = complex(v(4 + 2m - 1), v(4 + 2m))
        d1, d2, d3, d4 = complex(v(1)), complex(v(2)), complex(v(3)), complex(v(4))
        c12, c13, c14, c23, c24, c34 = c(1), c(2), c(3), c(4), c(5), c(6)
        # column-major
        N = SMatrix{4,4,ComplexF64,16}(d1, conj(c12), conj(c13), conj(c14),
                                       c12, d2, conj(c23), conj(c24),
                                       c13, c23, d3, conj(c34),
                                       c14, c24, c34, d4)
        return N, v(17)
    end
end

@inline function store_pol_state!(movie_pol, i, j, k, N, tauF)
    @inbounds begin
        for a in 1:4
            movie_pol[a, i, j, k] = real(N[a, a])
        end
        for (m, (a, b)) in enumerate(POL_UPPER)
            movie_pol[4 + 2m - 1, i, j, k] = real(N[a, b])
            movie_pol[4 + 2m, i, j, k] = imag(N[a, b])
        end
        movie_pol[17, i, j, k] = tauF
    end
    return nothing
end

"""
    render_round_cpu!(movie_nstep, movie_intensity, all_geodesics, valid_ks, target_times,
        pixels_x, pixels_y, params_slowlight, freq, model, simulation_data)

Render one round on the CPU: for every pixel and every currently-open
frame in `valid_ks`, continue integrating intensity along that pixel's
trajectory, threaded over pixels. Called from
[`process_slowlight_images!`](@ref) when `engine = :CPU`.

# Arguments
- `movie_nstep`: Remaining trajectory steps per pixel/frame, overwritten
  in place.
- `movie_intensity`: Accumulated intensity per pixel/frame, overwritten
  in place.
- `all_geodesics`: Matrix of pre-traced geodesic trajectories, one per
  pixel.
- `valid_ks`: Indices of the currently-active frames to render this
  round.
- `target_times`: Target simulation time for each frame.
- `pixels_x`, `pixels_y`: Image resolution.
- `params_slowlight`: Slow-light run state (time window `[tA, tB]`).
- `freq`: Frequency, in cgs units.
- `model`: Iharm model parameters.
- `simulation_data`: 3-element window of loaded GRMHD snapshots.
"""
function render_round_cpu!(
    movie_nstep, movie_intensity, all_geodesics, valid_ks, target_times, pixels_x, pixels_y,
    params_slowlight::OfSlowLight, freq, model, simulation_data;
    movie_pol = nothing
)
    Rh = 1 + sqrt(1.0 - model.a * model.a)
    dl_unit = model.L_unit * Constants.HPL / (Constants.ME * Constants.CL * Constants.CL)
    p = Progress(length(valid_ks) * pixels_x * pixels_y;
        desc = "Rendering $(length(valid_ks)) frame(s) this round on CPU...", showspeed = true, barlen = 30)
    progress_lock = ReentrantLock()

    Threads.@threads :greedy for i in 1:pixels_x
        for j in 1:pixels_y
            traj = all_geodesics[i, j]
            for k in valid_ks
                nstep = movie_nstep[i, j, k]
                Intensity = movie_intensity[i, j, k]
                dt = target_times[k] + 1e-5
                # Polarized frames carry N and tauF through the same steps, with the
                # coefficients taken at the same (shifted) Xf as Stokes I's jf, kf.
                N, tauF = movie_pol === nothing ? (Polarization.zero_tensor(Float64), 0.0) :
                          load_pol_state(movie_pol, i, j, k)

                while (nstep > 2)
                    Xi = traj[nstep].X
                    Xf = traj[nstep-1].X
                    Kconi = traj[nstep].Kcon
                    Kconf = traj[nstep-1].Kcon

                    Xi = SVector{4,Float64}(Xi[1] + dt, Xi[2], Xi[3], Xi[4])
                    Xf = SVector{4,Float64}(Xf[1] + dt, Xf[2], Xf[3], Xf[4])
                    if Xi[1] < params_slowlight.tA
                        shift = params_slowlight.tA - Xi[1]
                        Xf = SVector{4,Float64}(Xf[1] + shift, Xf[2], Xf[3], Xf[4])
                        Xi = SVector{4,Float64}(params_slowlight.tA, Xi[2], Xi[3], Xi[4])
                    end
                    if Xi[1] >= params_slowlight.tB
                        if Xf[1] >= params_slowlight.tf
                            shift = params_slowlight.tf - Xf[1]
                            Xi = SVector{4,Float64}(Xi[1] + shift, Xi[2], Xi[3], Xi[4])
                            Xf = SVector{4,Float64}(params_slowlight.tf, Xf[2], Xf[3], Xf[4])
                        else
                            break
                        end
                    end

                    ji, ki = Radiation.get_jk(Xi, Kconi, freq, model.a, model, simulation_data)
                    jf, kf = Radiation.get_jk(Xf, Kconf, freq, model.a, model, simulation_data)

                    Intensity = Radiation.approximate_solve(Intensity, ji, ki, jf, kf, traj[nstep-1].dl)

                    if movie_pol !== nothing
                        ti = traj[nstep]
                        N, tauF = Polarization.polarized_step(N, tauF, ti.X, Kconi, ti.Xhalf, ti.Kconhalf,
                            Xf, Kconf, traj[nstep-1].dl, dl_unit, Rh, freq, model.a, model, simulation_data)
                    end

                    nstep -= 1
                end
                movie_nstep[i, j, k] = nstep
                movie_intensity[i, j, k] = Intensity
                movie_pol === nothing || store_pol_state!(movie_pol, i, j, k, N, tauF)
            end
        end
        lock(progress_lock) do
            ProgressMeter.next!(p; step = pixels_y * length(valid_ks))
        end 
    end
    finish!(p)
    return nothing
end

"""
    pack_trajectory_tile!(dest, all_geodesics, nsteps, pixels_x, j0, j1)

Each pixel's traced light ray has its own number of steps, so
`all_geodesics` stores them as a list of differently-sized lists. The GPU
instead needs one same-sized rectangular block. This function builds that
block for one strip of the image (rows `j0` to `j1`): it copies each
pixel's steps into `dest`, padding any leftover space with zeros for
pixels whose ray was shorter than the longest one in the strip.

Only one strip at a time, not the whole image, to keep memory use down --
padding every pixel in the whole image to the single longest ray anywhere
could need tens of GB. Called from [`render_round_gpu!`](@ref), once per
strip per round.

# Arguments
- `dest`: Output buffer, overwritten with the packed trajectories.
- `all_geodesics`: Matrix of per-pixel geodesic trajectories.
- `nsteps`: Matrix of trajectory lengths, one per pixel.
- `pixels_x`: Image width.
- `j0`, `j1`: Range of image rows to pack.

# Returns
- `dest`.
"""
function pack_trajectory_tile!(dest, all_geodesics, nsteps, pixels_x, j0, j1)
    zero_vec = SVector{4,Float64}(0.0, 0.0, 0.0, 0.0)
    dummy = OfTrajGeneric{Float64}(0.0, zero_vec, zero_vec, zero_vec, zero_vec)
    fill!(dest, dummy)
    @inbounds for j in j0:j1, i in 1:pixels_x
        traj = all_geodesics[i, j]
        n = nsteps[i, j]
        jj = j - j0 + 1
        for s in 1:n
            dest[i, jj, s] = traj[s]
        end
    end
    return dest
end

"""
    upload_gpu_snapshot(cpu_snapshot)

Copy one CPU-resident GRMHD snapshot to the GPU. Small helper used by
[`init_gpu_data`](@ref) and [`refresh_gpu_data!`](@ref).

# Arguments
- `cpu_snapshot`: GRMHD snapshot with `Array`-backed fields.

# Returns
- The same snapshot, with `CuArray`-backed fields.
"""
upload_gpu_snapshot(cpu_snapshot) = Utils_GPU.copy_iharm_to_gpu(cpu_snapshot)

"""
    init_gpu_data(simulation_data)

Upload all 3 starting GRMHD snapshots to the GPU. Called once from
[`process_slowlight_images!`](@ref) when a `:GPU` run starts; after that,
[`refresh_gpu_data!`](@ref) takes over for each new round.

# Arguments
- `simulation_data`: 3-element window of loaded GRMHD snapshots.

# Returns
- 3-element `Vector` of GPU-resident snapshots.
"""
function init_gpu_data(simulation_data)
    return [upload_gpu_snapshot(simulation_data[1]),
            upload_gpu_snapshot(simulation_data[2]),
            upload_gpu_snapshot(simulation_data[3])]
end

"""
    refresh_gpu_data!(gpu_data, simulation_data)

Keep the GPU's copy of the GRMHD snapshots in sync after
[`update_data!`](@ref) slides the CPU-side window forward by one dump:
slots 1 and 2 just get re-pointed at the snapshots already on the GPU,
and only the genuinely new slot 3 is re-uploaded. Called from
[`process_slowlight_images!`](@ref) after each `update_data!` call, when
`engine = :GPU`.

# Arguments
- `gpu_data`: 3-element vector of GPU-resident snapshots, overwritten in
  place.
- `simulation_data`: 3-element window of loaded (CPU) GRMHD snapshots,
  already advanced by [`update_data!`](@ref).

# Returns
- `gpu_data`.
"""
function refresh_gpu_data!(gpu_data, simulation_data)
    gpu_data[1] = gpu_data[2]
    gpu_data[2] = gpu_data[3]
    gpu_data[3] = upload_gpu_snapshot(simulation_data[3])
    return gpu_data
end

"""
    process_slowlight_images!(params_slowlight, simulation_data, all_geodesics, nsteps, model, t0, tgeof, tgeoi, pixels_x, pixels_y, freq, Rhigh, all_dumps_path)

Render a slow-light movie: repeatedly integrate the (already-traced)
geodesics against the sliding GRMHD snapshot window, writing out one
image per `params_slowlight.ImageCadence` as the covered time window
allows, until every requested frame has been produced.

# Arguments
- `params_slowlight`: Slow-light run state.
- `simulation_data`: 3-element window of loaded GRMHD snapshots.
- `all_geodesics`: Matrix of pre-traced geodesic trajectories, one per
  pixel.
- `nsteps`: Matrix of trajectory lengths, one per pixel.
- `model`: Iharm model parameters.
- `t0`: Longest (most negative) photon travel time across all pixels.
- `tgeof`: Oldest simulation time needed by the active geodesics.
- `tgeoi`: Newest simulation time needed by the active geodesics.
- `pixels_x`, `pixels_y`: Image resolution.
- `freq`: Frequency, in cgs units.
- `Rhigh`: Electron/ion temperature ratio at high magnetization.
- `all_dumps_path`: `Printf`-style format string for the dump sequence.
- `Xcamera`: 4-vector camera position.
- `ro`, `theta_o`, `phi`: Camera position in KS spherical coordinates.
- `fovx`, `fovy`: Field of view, in radians.
- `SourceD`: Source distance, in cgs units.
- `scale`: Jy-per-pixel-intensity scale factor.
- `engine`: `:CPU` (default) integrates radiative transfer threaded over
  pixels on the CPU, exactly as before. `:GPU` runs the same physics on
  the GPU instead ([`render_round_gpu!`](@ref)/[`slowlight_kernel!`](@ref)).
"""
function process_slowlight_images!(
    params_slowlight, simulation_data, all_geodesics, nsteps,
    model, t0, tgeof, tgeoi, pixels_x, pixels_y, freq, Rhigh, all_dumps_path, Xcamera, ro, theta_o, phi, fovx, fovy, SourceD, scale;
    engine::Symbol = :CPU,
    polarized::Bool = false,
    output_dir::Union{Nothing,AbstractString} = nothing,
    t_ref::Real = NaN
)
    # With an explicit output_dir, frames are named by index, slow.00001.h5, ..., counted from the first
    # frame (camera time tA - tgeof) in steps of the cadence -- the numbering brisk and fast light use, so
    # frame k of every light mode is the same file number. Without it, the original Image.<time>.h5 names.
    # `t_ref` (band-0 modal emission time, from the brisk analysis of the same geometry) gives each frame's
    # n = 0 fluid time, t_camera + t_ref, recorded next to its camera time; NaN when not known.
    index_names = output_dir !== nothing
    if engine === :GPU && isnothing(Base.get_extension(parentmodule(@__MODULE__), :JipoleCUDAExt))
        error("engine = :GPU needs CUDA.jl loaded: run `using CUDA` first")
    end
    polarized && engine === :GPU && error("Polarized slow light is implemented for engine = :CPU only.")
    # Without an explicit output_dir, the original layout: ../slow_sims/<timestamp>.
    if output_dir === nothing
        base_dir = joinpath("..", "slow_sims")
        timestamp = Dates.format(now(), "yyyy-mm-dd-HH:MM:SS")
        output_dir = joinpath(base_dir, timestamp)
    end
    mkpath(output_dir)
    println("Outputs will be saved to: $output_dir")

    last_img_target = params_slowlight.tA - tgeof
    first_img_target = last_img_target
    nimgs_concurrently = round(Int, 2 + abs(t0) / params_slowlight.ImageCadence)

    movie_nstep = zeros(Int, pixels_x, pixels_y, nimgs_concurrently)
    movie_intensity = zeros(Float64, pixels_x, pixels_y, nimgs_concurrently)
    movie_pol = polarized ? zeros(Float64, POL_STATE_LEN, pixels_x, pixels_y, nimgs_concurrently) : nothing
    polarized && @printf("Polarized slow light: %d frames open at once, %.2f GB of polarization state\n",
        nimgs_concurrently, sizeof(movie_pol) / 1e9)
    target_times = zeros(Float64, nimgs_concurrently)
    valid_images = zeros(Float64, nimgs_concurrently)

    println("First Image will be produced at $last_img_target")
    nimg = 1
    nopenimgs = 1
    output = "Image.%07.1f.h5"

    max_nstep = 0
    tile_height = pixels_y
    gpu_data = nothing
    if engine === :GPU
        println("Preparing GPU workspace...")
        max_nstep = maximum(nsteps)
        tile_height = gpu_tile_plan(pixels_x, pixels_y, max_nstep, nimgs_concurrently, simulation_data)
        gpu_data = init_gpu_data(simulation_data)
        println("GPU tiling: $(cld(pixels_y, tile_height)) tile(s) of height $tile_height (image is $(pixels_x)x$(pixels_y), up to $max_nstep steps/pixel)")
    end

    while true
        while (last_img_target + t0 < params_slowlight.tB)
            target_times[nimg] = last_img_target
            if (last_img_target + tgeoi < params_slowlight.tf - model.rmax_geo)
                valid_images[nimg] = 1
                nopenimgs += 1
                for i in 1:pixels_x
                    for j in 1:pixels_y
                        movie_nstep[i, j, nimg] = nsteps[i, j]
                        movie_intensity[i, j, nimg] = 0.0
                        if movie_pol !== nothing
                            @views movie_pol[:, i, j, nimg] .= 0.0
                        end
                    end
                end
                nimg += 1
                if nimg > nimgs_concurrently
                    nimg = 1
                end
            end
            last_img_target += params_slowlight.ImageCadence
        end

        valid_ks = [k for k in 1:nimgs_concurrently if valid_images[k] == 1]

        elapsed = @elapsed if engine === :CPU
            render_round_cpu!(movie_nstep, movie_intensity, all_geodesics, valid_ks, target_times, pixels_x, pixels_y,
                params_slowlight, freq, model, simulation_data; movie_pol = movie_pol)
        else
            render_round_gpu!(movie_nstep, movie_intensity, all_geodesics, nsteps, max_nstep, tile_height, valid_ks,
                nimgs_concurrently, target_times, pixels_x, pixels_y, params_slowlight, freq, model, gpu_data)
        end
        println("Round rendered in $(round(elapsed, digits=3))s on $engine")

        # do_output is derived from movie_nstep's final state, so it's identical
        # regardless of which engine produced that state.
        do_output = trues(nimgs_concurrently)
        for k in valid_ks
            if !all(==(2), @view movie_nstep[:, :, k])
                do_output[k] = false
            end
        end

        for k in valid_ks
            if do_output[k]
                Image_out = movie_intensity[:, :, k] .* freq^3

                frame_index = round(Int, (target_times[k] - first_img_target) / params_slowlight.ImageCadence) + 1
                file_name = index_names ? joinpath(output_dir, @sprintf("slow.%05d.h5", frame_index)) :
                            joinpath(output_dir, Printf.format(Printf.Format(output), target_times[k]))
                out_data = Dict{String, Any}(
                    "image"      => Image_out,
                    "img_time"   => target_times[k],
                    "params"     => model,                
                    "data"       => simulation_data[1],   
                    "ro"         => ro,           
                    "theta_o"    => theta_o,       
                    "phi"        => phi,         
                    "fovx"       => fovx,           
                    "fovy"       => fovy,           
                    "freq"       => freq,                 
                    "SourceD"    => SourceD,        
                    "scale"      => scale,          
                    "Xcamera"    => Xcamera,        
                    "Rhigh" => Rhigh,
                    "frame_index" => frame_index,
                    "t_camera" => target_times[k],
                    "t_fluid" => target_times[k] + t_ref
                )
                if movie_pol !== nothing
                    pol = zeros(Float64, Polarization.NIMG, pixels_x, pixels_y)
                    Threads.@threads for i in 1:pixels_x
                        for j in 1:pixels_y
                            N, tauF = load_pol_state(movie_pol, i, j, k)
                            SI, SQ, SU, SV = Polarization.project_n(N, all_geodesics[i, j][1].X, model.a, model)
                            Polarization.save_pixel!(pol, i, j, SI, SQ, SU, SV, tauF, freq, 0)
                        end
                    end
                    out_data["pol"] = pol
                end
                Output.generate_output_file(file_name, out_data; format="ipole")
                println("Saving image $(file_name)")

                valid_images[k] = 0
                nopenimgs -= 1
            end
        end

        if nopenimgs <= 1
            break
        end
        update_data!(params_slowlight, simulation_data, Rhigh, model, all_dumps_path)
        if engine === :GPU
            refresh_gpu_data!(gpu_data, simulation_data)
        end
    end
end


end
