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
using CUDA
using ..Constants
using ..Radiation
using ..Iharm
using ..Output
using ..GeoTypes
using ..Utils_GPU
using Dates
export OfSlowLight, update_dump_path, get_specific_dump_time, update_data!,
    process_slowlight_images!

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
    update_data!(params_slowlight, simulation_data, trat_large, model, all_dumps_path)

Slide the 3-snapshot window `simulation_data` forward by one dump,
loading the next dump into the (reused) oldest slot.

# Arguments
- `params_slowlight`: Slow-light run state, updated with the new time
  window `[tA, tB]`.
- `simulation_data`: 3-element window of loaded GRMHD snapshots.
- `trat_large`: Electron/ion temperature ratio at high magnetization.
- `model`: Iharm model parameters.
- `all_dumps_path`: `Printf`-style format string for the dump sequence.
"""
function update_data!(params_slowlight::OfSlowLight, simulation_data, trat_large::Float64, model::Iharm.IharmParams, all_dumps_path::String)
    oldest_data = simulation_data[1]

    simulation_data[1] = simulation_data[2]
    simulation_data[2] = simulation_data[3]

    simulation_data[3] = oldest_data

    simulation_data[3] = Iharm.load_data(params_slowlight.current_dumps_path, trat_large, model;
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

            lock(progress_lock) do
                for _ in valid_ks
                    ProgressMeter.next!(p)
                end
            end
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
    dummy = OfTrajS(0.0, zero_vec, zero_vec, zero_vec, zero_vec)
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
    gpu_tile_plan(pixels_x, pixels_y, max_nstep, nimgs_concurrently, simulation_data)

Decide how many image rows fit in one GPU tile, by checking how much
memory is actually free right now -- on both the GPU (`CUDA.available_memory()`)
and the host (`Sys.free_memory()`, since [`pack_trajectory_tile!`](@ref)
builds each tile on the host first) -- and picking the size that fits
the tighter of the two. Called once from [`process_slowlight_images!`](@ref)
when `engine = :GPU`, before the rounds loop starts.

# Arguments
- `pixels_x`, `pixels_y`: Image resolution.
- `max_nstep`: Longest trajectory length across all pixels.
- `nimgs_concurrently`: Number of frames rendered concurrently.
- `simulation_data`: 3-element window of loaded GRMHD snapshots, used to
  estimate their GPU memory footprint.

# Returns
- `tile_height`: number of image rows (the `j` dimension) per tile.
"""
function gpu_tile_plan(pixels_x, pixels_y, max_nstep, nimgs_concurrently, simulation_data)
    traj_elem_bytes = sizeof(OfTrajS)

    grid_bytes = 3 * Base.summarysize(simulation_data[1])
    state_bytes = pixels_x * pixels_y * nimgs_concurrently * (sizeof(Float64) + sizeof(Int))

    gpu_usable_bytes = max(CUDA.available_memory() - grid_bytes - state_bytes, 0)
    host_usable_bytes = Sys.free_memory()

    safety_fraction = 0.6 
    usable_bytes = floor(Int, min(gpu_usable_bytes, host_usable_bytes) * safety_fraction)

    row_bytes = pixels_x * max_nstep * traj_elem_bytes
    tile_height = row_bytes <= 0 ? pixels_y : clamp(usable_bytes ÷ row_bytes, 1, pixels_y)
    return tile_height
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
    slowlight_kernel!(traj, nstep_state, intensity, valid_mask, target_times,
        tA, tB, tf, freq, bhspin, model, data)

The GPU kernel itself: one thread per pixel `(i, j)`, looping over every
active frame `k` (skipping any where `valid_mask[k] == 0`) and continuing
that pixel's intensity integration along its trajectory. Same math as
[`render_round_cpu!`](@ref)'s inner loop, just run on the GPU. Launched
via `@cuda` from [`render_round_gpu!`](@ref).

# Arguments
- `traj`: Packed per-pixel geodesic trajectories for this tile (see
  [`pack_trajectory_tile!`](@ref)).
- `nstep_state`: Remaining trajectory steps per pixel/frame, overwritten
  in place.
- `intensity`: Accumulated intensity per pixel/frame, overwritten in
  place.
- `valid_mask`: Which frames are currently active (nonzero entries).
- `target_times`: Target simulation time for each frame.
- `tA`, `tB`: Time window currently bracketed by `data`.
- `tf`: Simulation time of the newest available dump.
- `freq`: Frequency, in cgs units.
- `bhspin`: Dimensionless black hole spin parameter.
- `model`: Iharm model parameters.
- `data`: GPU-resident, 3-snapshot `NTuple` of GRMHD snapshots (see
  [`refresh_gpu_data!`](@ref)).
"""
function slowlight_kernel!(
    traj, nstep_state, intensity, valid_mask, target_times,
    tA::Float64, tB::Float64, tf::Float64,
    freq::Float64, bhspin::Float64, model, data
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    nx, ny, _ = size(traj)
    nk = length(valid_mask)

    if i <= nx && j <= ny
        @inbounds for k in 1:nk
            if valid_mask[k] == 0
                continue
            end

            nstep = nstep_state[i, j, k]
            Intensity = intensity[i, j, k]
            dt = target_times[k] + 1e-5

            while nstep > 2
                Xi = traj[i, j, nstep].X
                Xf = traj[i, j, nstep-1].X
                Kconi = traj[i, j, nstep].Kcon
                Kconf = traj[i, j, nstep-1].Kcon

                Xi = SVector{4,Float64}(Xi[1] + dt, Xi[2], Xi[3], Xi[4])
                Xf = SVector{4,Float64}(Xf[1] + dt, Xf[2], Xf[3], Xf[4])
                if Xi[1] < tA
                    shift = tA - Xi[1]
                    Xf = SVector{4,Float64}(Xf[1] + shift, Xf[2], Xf[3], Xf[4])
                    Xi = SVector{4,Float64}(tA, Xi[2], Xi[3], Xi[4])
                end
                if Xi[1] >= tB
                    if Xf[1] >= tf
                        shift = tf - Xf[1]
                        Xi = SVector{4,Float64}(Xi[1] + shift, Xi[2], Xi[3], Xi[4])
                        Xf = SVector{4,Float64}(tf, Xf[2], Xf[3], Xf[4])
                    else
                        break
                    end
                end

                ji, ki = Radiation.get_jk(Xi, Kconi, freq, bhspin, model, data)
                jf, kf = Radiation.get_jk(Xf, Kconf, freq, bhspin, model, data)

                Intensity = Radiation.approximate_solve(Intensity, ji, ki, jf, kf, traj[i, j, nstep-1].dl)

                nstep -= 1
            end

            nstep_state[i, j, k] = nstep
            intensity[i, j, k] = Intensity
        end
    end
    return nothing
end

"""
    render_round_gpu!(movie_nstep, movie_intensity, all_geodesics, nsteps, max_nstep, tile_height,
        valid_ks, nimgs_concurrently, target_times, pixels_x, pixels_y, params_slowlight, freq, model, gpu_data)

Render one round on the GPU -- the `:GPU` counterpart of
[`render_round_cpu!`](@ref). Goes tile by tile (rows sized by
[`gpu_tile_plan`](@ref)): pack that tile's trajectories
([`pack_trajectory_tile!`](@ref)) and upload it plus the current
`movie_nstep`/`movie_intensity` state, run [`slowlight_kernel!`](@ref),
then copy the results back to `movie_nstep`/`movie_intensity` so they're
plain host arrays again afterwards, same as the CPU path. Called from
[`process_slowlight_images!`](@ref) when `engine = :GPU`.

# Arguments
- `movie_nstep`: Remaining trajectory steps per pixel/frame, overwritten
  in place.
- `movie_intensity`: Accumulated intensity per pixel/frame, overwritten
  in place.
- `all_geodesics`: Matrix of pre-traced geodesic trajectories, one per
  pixel.
- `nsteps`: Matrix of trajectory lengths, one per pixel.
- `max_nstep`: Longest trajectory length across all pixels.
- `tile_height`: Number of image rows per tile (see [`gpu_tile_plan`](@ref)).
- `valid_ks`: Indices of the currently-active frames to render this
  round.
- `nimgs_concurrently`: Number of frames rendered concurrently.
- `target_times`: Target simulation time for each frame.
- `pixels_x`, `pixels_y`: Image resolution.
- `params_slowlight`: Slow-light run state (time window `[tA, tB, tf]`).
- `freq`: Frequency, in cgs units.
- `model`: Iharm model parameters.
- `gpu_data`: 3-element vector of GPU-resident GRMHD snapshots.
"""
function render_round_gpu!(
    movie_nstep, movie_intensity, all_geodesics, nsteps, max_nstep, tile_height, valid_ks, nimgs_concurrently,
    target_times, pixels_x, pixels_y, params_slowlight::OfSlowLight, freq, model, gpu_data
)
    threads_per_block = (16, 16)
    valid_mask_host = zeros(Int, nimgs_concurrently)
    for k in valid_ks
        valid_mask_host[k] = 1
    end
    d_valid_mask = CuArray(valid_mask_host)
    d_target_times = CuArray(target_times)
    data_tuple = Tuple(gpu_data)

    n_tiles = cld(pixels_y, tile_height)
    println("Rendering $(length(valid_ks)) frame(s) this round on GPU ($n_tiles tile(s) of height $tile_height)...")

    traj_tile_host = Array{OfTrajS}(undef, pixels_x, tile_height, max_nstep)
    for j0 in 1:tile_height:pixels_y
        j1 = min(j0 + tile_height - 1, pixels_y)
        tile_ny = j1 - j0 + 1

        traj_view = tile_ny == tile_height ? traj_tile_host : Array{OfTrajS}(undef, pixels_x, tile_ny, max_nstep)
        pack_trajectory_tile!(traj_view, all_geodesics, nsteps, pixels_x, j0, j1)
        d_traj = CuArray(traj_view)

        d_nstep = CuArray(@view movie_nstep[:, j0:j1, :])
        d_intensity = CuArray(@view movie_intensity[:, j0:j1, :])

        blocks_per_grid = (cld(pixels_x, threads_per_block[1]), cld(tile_ny, threads_per_block[2]))
        @cuda threads = threads_per_block blocks = blocks_per_grid slowlight_kernel!(
            d_traj, d_nstep, d_intensity, d_valid_mask, d_target_times,
            params_slowlight.tA, params_slowlight.tB, params_slowlight.tf,
            freq, model.a, model, data_tuple
        )
        CUDA.synchronize()

        copyto!(@view(movie_nstep[:, j0:j1, :]), Array(d_nstep))
        copyto!(@view(movie_intensity[:, j0:j1, :]), Array(d_intensity))
    end
    return nothing
end

"""
    process_slowlight_images!(params_slowlight, simulation_data, all_geodesics, nsteps, model, t0, tgeof, tgeoi, pixels_x, pixels_y, freq, trat_large, all_dumps_path)

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
- `trat_large`: Electron/ion temperature ratio at high magnetization.
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
    model, t0, tgeof, tgeoi, pixels_x, pixels_y, freq, trat_large, all_dumps_path, Xcamera, ro, theta_o, phi, fovx, fovy, SourceD, scale;
    engine::Symbol = :CPU
)
    engine in (:CPU, :GPU) || throw(ArgumentError("engine must be :CPU or :GPU, got $(repr(engine))"))

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
                    "trat_large" => trat_large            
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
        update_data!(params_slowlight, simulation_data, trat_large, model, all_dumps_path)
        if engine === :GPU
            refresh_gpu_data!(gpu_data, simulation_data)
        end
    end
end


end
