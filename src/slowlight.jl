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
using ..Output
using ..GeoTypes
using ..Geodesics
using ..Utils_GPU
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
    t::Float64 = 0.0
    h5open(dump_path, "r") do file
        t = read(file, "t")
    end
    return t
end

"""
    update_data!(params_slowlight, simulation_data, Rhigh, model, all_dumps_path)

Slide the 3-snapshot window `simulation_data` forward by one dump,
loading the next dump into the (reused) oldest slot.

# Arguments
- `params_slowlight`: Slow-light run state, updated with the new time
  window `[tA, tB]`.
- `simulation_data`: 3-element window of loaded GRMHD snapshots.
- `Rhigh`: Electron/ion temperature ratio at high magnetization.
- `model`: Iharm model parameters.
- `all_dumps_path`: `Printf`-style format string for the dump sequence.
"""
function update_data!(params_slowlight::OfSlowLight, simulation_data, Rhigh::Float64, model::Iharm.IharmParams, all_dumps_path::String)
    oldest_data = simulation_data[1]

    simulation_data[1] = simulation_data[2]
    simulation_data[2] = simulation_data[3]

    simulation_data[3] = oldest_data

    simulation_data[3] = Iharm.load_data(params_slowlight.current_dumps_path, Rhigh, model;
        advance_path! = () -> (params_slowlight.current_dumps_path = update_dump_path(params_slowlight, all_dumps_path)))

    params_slowlight.tA = simulation_data[1].t
    params_slowlight.tB = simulation_data[2].t

    @info "Loaded data" dump_path = params_slowlight.current_dumps_path tA = params_slowlight.tA tB = params_slowlight.tB
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
    params_slowlight::OfSlowLight, freq, model, simulation_data
)
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

                    nstep -= 1
                end
                movie_nstep[i, j, k] = nstep
                movie_intensity[i, j, k] = Intensity
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
    trace_slowlight(model, Xcam, Econ, fovx, fovy, freq_unitless, pixels_x, pixels_y, maxnstep, Rh, xoff, yoff)

Trace the geodesic of every pixel once, for the whole slow-light run (geodesics do not
depend on the dump), and find the time range the run needs, as ipole does: `t0`, the
earliest time any geodesic reaches, and `tgeoi`/`tgeof`, the newest and oldest times at
which the geodesics are inside r = 100. Used by `scripts/main.jl` and the slow-light notebook.

# Returns
- `(all_geodesics, nsteps, midplane_crossings, t0, tgeof, tgeoi)`: each pixel's trajectory
  (trimmed to its length), its number of points and of midplane crossings, and the three times.
"""
function trace_slowlight(model, Xcam, Econ, fovx, fovy, freq_unitless, pixels_x, pixels_y, maxnstep, Rh, xoff, yoff)
    dummy_svec = zero(SVector{4,Float64})
    dummy_traj = OfTrajGeneric{Float64}(0.0, dummy_svec, dummy_svec, dummy_svec, dummy_svec)

    nbuf = Threads.nthreads()
    traj_pool = Channel{Vector{OfTrajGeneric{Float64}}}(nbuf)
    for _ in 1:nbuf
        put!(traj_pool, fill(dummy_traj, maxnstep))
    end

    all_geodesics = Matrix{Vector{OfTrajGeneric{Float64}}}(undef, pixels_x, pixels_y)
    nsteps = zeros(Int, pixels_x, pixels_y)
    midplane_crossings = zeros(Int, pixels_x, pixels_y)

    row_t0 = zeros(Float64, pixels_x)
    row_tgeoi = fill(-1e100, pixels_x)
    row_tgeof = zeros(Float64, pixels_x)

    p = Progress(pixels_x * pixels_y; desc="Tracing geodesics...", showspeed=true, barlen=30)
    progress_lock = ReentrantLock()
    Threads.@threads :greedy for i in 0:(pixels_x - 1)
        my_traj = take!(traj_pool)
        for j in 0:(pixels_y - 1)
            nstep, crossings = Geodesics.get_pixel(my_traj, i, j, Xcam, Econ, fovx, fovy, freq_unitless, pixels_x, pixels_y, model.a, Rh, model.rmax_geo, model, xoff, yoff)

            nsteps[i + 1, j + 1] = nstep
            midplane_crossings[i + 1, j + 1] = crossings
            all_geodesics[i + 1, j + 1] = my_traj[1:nstep]

            final_step_time = my_traj[nstep].X[1]
            if final_step_time < row_t0[i + 1]
                row_t0[i + 1] = final_step_time
            end

            pixel_tgeoi = 1.0
            pixel_tgeof = 1.0
            for k in 1:nstep
                X = my_traj[k].X
                K = my_traj[k].Kcon

                log_r = X[2]
                t_coord = X[1]
                k_r = K[2]

                if pixel_tgeoi > 0.0 && log_r < log(100.0)
                    pixel_tgeoi = t_coord
                end
                if pixel_tgeof > 0.0 && log_r > log(100.0) && k_r < 0.0
                    pixel_tgeof = t_coord
                end
            end

            if pixel_tgeoi < 0.0 && pixel_tgeoi > row_tgeoi[i + 1]
                row_tgeoi[i + 1] = pixel_tgeoi
            end
            if pixel_tgeof < 0.0 && pixel_tgeof < row_tgeof[i + 1]
                row_tgeof[i + 1] = pixel_tgeof
            elseif pixel_tgeof > 0.0 && final_step_time < row_tgeof[i + 1]
                row_tgeof[i + 1] = final_step_time
            end
        end
        lock(progress_lock) do
            ProgressMeter.next!(p; step = pixels_y)
        end
        put!(traj_pool, my_traj)
    end
    finish!(p)

    return all_geodesics, nsteps, midplane_crossings, minimum(row_t0), minimum(row_tgeof), maximum(row_tgeoi)
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
    engine::Symbol = :CPU
)
    if engine === :GPU && isnothing(Base.get_extension(parentmodule(@__MODULE__), :JipoleCUDAExt))
        error("engine = :GPU needs CUDA.jl loaded: run `using CUDA` first")
    end
    base_dir = joinpath("..", "slow_sims")
    
    if !isdir(base_dir)
        mkpath(base_dir)
    end
    

    timestamp = Dates.format(now(), "yyyy-mm-dd-HH:MM:SS")
    
    output_dir = joinpath(base_dir, timestamp)
    mkpath(output_dir)
    println("Outputs will be saved to: $output_dir")

    last_img_target = params_slowlight.tA - tgeof
    nimgs_concurrently = round(Int, 2 + abs(t0) / params_slowlight.ImageCadence)

    movie_nstep = zeros(Int, pixels_x, pixels_y, nimgs_concurrently)
    movie_intensity = zeros(Float64, pixels_x, pixels_y, nimgs_concurrently)
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
        gpu_data = [Utils_GPU.copy_iharm_to_gpu(d) for d in simulation_data]
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
                params_slowlight, freq, model, simulation_data)
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

                file_name = joinpath(output_dir, Printf.format(Printf.Format(output), target_times[k]))
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
                    "Rhigh" => Rhigh            
                )
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
            gpu_data[1], gpu_data[2], gpu_data[3] = gpu_data[2], gpu_data[3], Utils_GPU.copy_iharm_to_gpu(simulation_data[3])
        end
    end
end


end
