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
using ..Polarization
using ..Utils_GPU
using Dates
export OfSlowLight, update_dump_path, get_specific_dump_time, update_data!,
    process_slowlight_images!


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
      render_round_cpu!(movie_nstep, movie_intensity, movie_N, movie_tauF, all_geodesics, valid_ks, target_times,
          pixels_x, pixels_y, params_slowlight, freq, model, simulation_data)

  Render one round on the CPU: for every pixel and every currently-open
  frame in `valid_ks`, continue integrating intensity along that pixel's
  trajectory, threaded over pixels. With `movie_N`/`movie_tauF` arrays (polarized
  run) the coherency tensor and Faraday depth are carried along too, as in
  ipole; with `nothing` the polarized code is not compiled in. Called from
  [`process_slowlight_images!`](@ref) when `engine = :CPU`.

  # Arguments
  - `movie_nstep`: Remaining trajectory steps per pixel/frame, overwritten in place.
  - `movie_intensity`: Accumulated intensity per pixel/frame, overwritten in place.
  - `movie_N`, `movie_tauF`: Coherency tensor and Faraday depth per pixel/frame,
    overwritten in place, or `nothing` for an unpolarized run.
  - `all_geodesics`: Matrix of pre-traced geodesic trajectories, one per pixel.
  - `valid_ks`: Indices of the currently-active frames to render this round.
  - `target_times`: Target simulation time for each frame.
  - `pixels_x`, `pixels_y`: Image resolution.
  - `params_slowlight`: Slow-light run state (time window `[tA, tB]`).
  - `freq`: Frequency, in cgs units.
  - `model`: Iharm model parameters.
  - `simulation_data`: 3-element window of loaded GRMHD snapshots.
"""
function render_round_cpu!(
      movie_nstep, movie_intensity, movie_N, movie_tauF, all_geodesics, valid_ks, target_times, pixels_x, pixels_y,
      params_slowlight::OfSlowLight, freq, model, simulation_data
  )
      polarized = movie_N !== nothing
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
                  N = polarized ? movie_N[i, j, k] : Polarization.zero_tensor(Float64)
                  tauF = polarized ? movie_tauF[i, j, k] : 0.0
                  dt = target_times[k] + 1e-5

                  while (nstep > 2)
                      Xi = traj[nstep].X
                      Xf = traj[nstep-1].X
                      Xh = traj[nstep].Xhalf
                      Kconi = traj[nstep].Kcon
                      Kconf = traj[nstep-1].Kcon

                      # Move the step to the frame's time; the half-step point moves with it (as in ipole).
                      Xi = SVector{4,Float64}(Xi[1] + dt, Xi[2], Xi[3], Xi[4])
                      Xf = SVector{4,Float64}(Xf[1] + dt, Xf[2], Xf[3], Xf[4])
                      Xh = SVector{4,Float64}(Xh[1] + dt, Xh[2], Xh[3], Xh[4])
                      if Xi[1] < params_slowlight.tA
                          shift = params_slowlight.tA - Xi[1]
                          Xf = SVector{4,Float64}(Xf[1] + shift, Xf[2], Xf[3], Xf[4])
                          Xh = SVector{4,Float64}(Xh[1] + shift, Xh[2], Xh[3], Xh[4])
                          Xi = SVector{4,Float64}(params_slowlight.tA, Xi[2], Xi[3], Xi[4])
                      end
                      if Xi[1] >= params_slowlight.tB
                        if Xf[1] >= params_slowlight.tf
                            shift = params_slowlight.tf - Xf[1]
                            Xi = SVector{4,Float64}(Xi[1] + shift, Xi[2], Xi[3], Xi[4])
                            Xh = SVector{4,Float64}(Xh[1] + shift, Xh[2], Xh[3], Xh[4])
                            Xf = SVector{4,Float64}(params_slowlight.tf, Xf[2], Xf[3], Xf[4])
                        else
                            break
                        end
                    end

                    ji, ki = Radiation.get_jk(Xi, Kconi, freq, model.a, model, simulation_data)
                    jf, kf = Radiation.get_jk(Xf, Kconf, freq, model.a, model, simulation_data)

                    Intensity = Radiation.approximate_solve(Intensity, ji, ki, jf, kf, traj[nstep-1].dl)

                    if polarized
                        N, tauF = Polarization.polarized_step(N, tauF, Xi, Kconi, Xh, traj[nstep].Kconhalf, Xf, Kconf,
                            traj[nstep-1].dl, dl_unit, Rh, freq, model.a, model, simulation_data)
                    end

                    nstep -= 1
                end
                movie_nstep[i, j, k] = nstep
                movie_intensity[i, j, k] = Intensity
                if polarized
                    movie_N[i, j, k] = N
                    movie_tauF[i, j, k] = tauF
                end
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
    flatten_rays(all_geodesics, nsteps)

Put all rays back to back in one flat array, each at its own length, for the GPU.
Pixels go in memory order (i fastest), so a block of image rows `j0:j1` is one
contiguous slice. Padding every ray to the longest one instead took about 4.5 times
more memory. Called once from [`process_slowlight_images!`](@ref) when `engine = :GPU`.

# Arguments
- `all_geodesics`: Matrix of per-pixel geodesic trajectories.
- `nsteps`: Matrix of trajectory lengths, one per pixel.

# Returns
- `(rays, offsets)`: the flat array, and where each pixel's ray starts in it
(point `s` of pixel `(i, j)` is `rays[offsets[i, j] + s]`).
"""
function flatten_rays(all_geodesics, nsteps)
    offsets = similar(nsteps)
    total = 0
    for j in axes(nsteps, 2), i in axes(nsteps, 1)
        offsets[i, j] = total
        total += nsteps[i, j]
    end
    rays = Vector{eltype(all_geodesics[1])}(undef, total)
    for j in axes(nsteps, 2), i in axes(nsteps, 1)
        copyto!(rays, offsets[i, j] + 1, all_geodesics[i, j], 1, nsteps[i, j])
    end
    return rays, offsets
end


"""
    shifted_points(rays, off, nstep, dt, tA, tB, tf)

The three points of one step of a stored ray (far end `nstep`, its half-step point, near
end `nstep - 1`), moved to a frame's time (`dt` after the ray's own time) and clamped to
the current data window, as in ipole (main.c:388–475): a step that starts before `tA` is
moved up to `tA`, and one past `tB` either ends at `tf` or waits for the next round.
Used by both passes of [`slowlight_kernel!`](@ref).

# Arguments
- `rays`: Stored ray points; point `s` of this ray is `rays[off + s]`.
- `off`: Where this ray starts in `rays`.
- `nstep`: The step's far end.
- `dt`: The frame's target time plus `1e-5`.
- `tA`, `tB`: Time window currently bracketed by the loaded dumps.
- `tf`: Simulation time of the newest available dump.

# Returns
- `(Xi, Xh, Xf, stop)`: far end, half-step point and near end at the frame's time, and
whether the step is past the window (the frame then stops here this round).
"""
@inline function shifted_points(rays, off, nstep, dt, tA, tB, tf)
    @inbounds Xi = rays[off + nstep].X
    @inbounds Xf = rays[off + nstep - 1].X
    @inbounds Xh = rays[off + nstep].Xhalf
    Xi = SVector{4,Float64}(Xi[1] + dt, Xi[2], Xi[3], Xi[4])
    Xf = SVector{4,Float64}(Xf[1] + dt, Xf[2], Xf[3], Xf[4])
    Xh = SVector{4,Float64}(Xh[1] + dt, Xh[2], Xh[3], Xh[4])
    if Xi[1] < tA
        shift = tA - Xi[1]
        Xf = SVector{4,Float64}(Xf[1] + shift, Xf[2], Xf[3], Xf[4])
        Xh = SVector{4,Float64}(Xh[1] + shift, Xh[2], Xh[3], Xh[4])
        Xi = SVector{4,Float64}(tA, Xi[2], Xi[3], Xi[4])
    end
    if Xi[1] >= tB
        if Xf[1] >= tf
            shift = tf - Xf[1]
            Xi = SVector{4,Float64}(Xi[1] + shift, Xi[2], Xi[3], Xi[4])
            Xh = SVector{4,Float64}(Xh[1] + shift, Xh[2], Xh[3], Xh[4])
            Xf = SVector{4,Float64}(tf, Xf[2], Xf[3], Xf[4])
        else
            return Xi, Xh, Xf, true
        end
    end     
    return Xi, Xh, Xf, false
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
- `pol`: also render the polarized frames (Stokes I, Q, U, V and Faraday depth).
- `qu_conv`: Q/U sign convention of the polarized frames, see `Polarization.save_pixel!`.
"""
function process_slowlight_images!(
    params_slowlight, simulation_data, all_geodesics, nsteps,
    model, t0, tgeof, tgeoi, pixels_x, pixels_y, freq, Rhigh, all_dumps_path, Xcamera, ro, theta_o, phi, fovx, fovy, SourceD, scale;
    engine::Symbol = :CPU, pol::Bool = false, qu_conv::Int = 0
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
    # For polarization
    movie_N = pol ? fill(Polarization.zero_tensor(Float64), pixels_x, pixels_y, nimgs_concurrently) : nothing
    movie_tauF = pol ? zeros(Float64, pixels_x, pixels_y, nimgs_concurrently) : nothing
    pol && println("Polarized frame state: $(round((sizeof(movie_N) + sizeof(movie_tauF)) / 1024^3, digits=2)) GB")
    #
    target_times = zeros(Float64, nimgs_concurrently)
    valid_images = zeros(Float64, nimgs_concurrently)

    println("First Image will be produced at $last_img_target")
    nimg = 1
    nopenimgs = 1
    output = "Image.%07.1f.h5"

    rays, offsets = nothing, nothing
    tile_height = pixels_y
    gpu_data = nothing
    if engine === :GPU
        println("Preparing GPU workspace...")
        rays, offsets = flatten_rays(all_geodesics, nsteps)
        # Rows per tile: a row's stored rays must fit next to the 3 snapshots and the frame state.
        state_bytes = sizeof(movie_nstep) + sizeof(movie_intensity) + (pol ? sizeof(movie_N) + sizeof(movie_tauF) : 0)
        tile_height = min(pixels_y, Utils_GPU.gpu_tile_plan(maximum(sum(nsteps; dims = 1)) * sizeof(eltype(rays));
        reserve_bytes = 3 * Base.summarysize(simulation_data[1]) + state_bytes))
        # When all rays fit on the GPU, upload them once here and keep them for the whole run.
        tile_height == pixels_y && (rays = Utils_GPU.copy_iharm_to_gpu(rays))
        gpu_data = [Utils_GPU.copy_iharm_to_gpu(d) for d in simulation_data]
        println("GPU tiling: $(cld(pixels_y, tile_height)) tile(s) of height $tile_height ($(round(sizeof(rays) / 1e9, digits=2)) GB of rays)")
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
                if movie_N !== nothing
                    fill!(view(movie_N, :, :, nimg), Polarization.zero_tensor(Float64))
                    fill!(view(movie_tauF, :, :, nimg), 0.0)
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
            render_round_cpu!(movie_nstep, movie_intensity, movie_N, movie_tauF, all_geodesics, valid_ks, target_times, pixels_x, pixels_y,
                params_slowlight, freq, model, simulation_data)
        else
             render_round_gpu!(movie_nstep, movie_intensity, movie_N, movie_tauF, rays, offsets,
                tile_height, valid_ks, target_times, params_slowlight, freq, model, gpu_data)
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
                if movie_N !== nothing
                    # Polarized frame: project each pixel's tensor where its integration stopped, one step
                    # short of the camera, as ipole does (project_N at the last point, main.c:467).
                    pol_out = zeros(Polarization.NIMG, pixels_x, pixels_y)
                    for i in 1:pixels_x, j in 1:pixels_y
                        X_end = all_geodesics[i, j][movie_nstep[i, j, k]].X
                        SI, SQ, SU, SV = Polarization.project_n(movie_N[i, j, k], X_end, model.a, model)
                        Polarization.save_pixel!(pol_out, i, j, SI, SQ, SU, SV, movie_tauF[i, j, k], freq, qu_conv)
                    end
                    out_data["pol"] = pol_out
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
            gpu_data[1], gpu_data[2], gpu_data[3] = gpu_data[2], gpu_data[3], Utils_GPU.copy_iharm_to_gpu(simulation_data[3])
        end
    end
end


end
