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

export OfImg, OfSlowLight, update_dump_path, get_specific_dump_time, update_data!,
    process_slowlight_images!

"""
Per-pixel, per-frame accumulator used while rendering a slow-light movie.
"""
mutable struct OfImg
    nstep::Int
    Intensity::Float64
    tau::Float64
    tauF::Float64
    N_coord::MMatrix{4,4,ComplexF64}
end

"""
    Base.zero(::Type{OfImg})

Construct a zeroed [`OfImg`](@ref).
"""
function Base.zero(::Type{OfImg})
    OfImg(
        0,
        0.0,
        0.0,
        0.0,
        MMatrix{4,4,ComplexF64}(zeros(ComplexF64, 4, 4))
    )
end

"""
Slow-light run state: which dump is currently loaded/being advanced to,
and the time window `[tA, tB]` currently bracketed by `simulation_data`.
"""
mutable struct OfSlowLight
    dump_max::Int64
    nloaded::Int64
    ImageCadence::Int64
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
"""
function process_slowlight_images!(
    params_slowlight, simulation_data, all_geodesics, nsteps,
    model, t0, tgeof, tgeoi, pixels_x, pixels_y, freq, trat_large, all_dumps_path
)
    last_img_target = params_slowlight.tA - tgeof
    nimgs_concurrently = round(Int, 2 + abs(t0) / params_slowlight.ImageCadence)

    MovieArray = [zero(OfImg) for _ in 1:pixels_x, _ in 1:pixels_y, _ in 1:nimgs_concurrently]
    target_times = zeros(Float64, nimgs_concurrently)
    valid_images = zeros(Float64, nimgs_concurrently)

    println("First Image will be produced at $last_img_target")
    nimg = 1
    nopenimgs = 1
    output = "Image.%05d.txt"

    while true
        while (last_img_target + t0 < params_slowlight.tB)
            target_times[nimg] = last_img_target
            if (last_img_target + tgeoi < params_slowlight.tf - model.rmax_geo)
                valid_images[nimg] = 1
                nopenimgs += 1
                for i in 1:pixels_x
                    for j in 1:pixels_y
                        MovieArray[i, j, nimg].nstep = nsteps[i, j]
                        MovieArray[i, j, nimg].Intensity = 0.0
                        MovieArray[i, j, nimg].tau = 0.0
                        MovieArray[i, j, nimg].tauF = 0.0
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
        p = Progress(length(valid_ks) * pixels_x * pixels_y;
            desc = "Rendering $(length(valid_ks)) frame(s) this round...", showspeed = true, barlen = 30)
        progress_lock = ReentrantLock()
        for k in valid_ks
            do_output = true

            Threads.@threads :greedy for i in 1:pixels_x
                for j in 1:pixels_y
                    traj = all_geodesics[i, j]
                    nstep = MovieArray[i, j, k].nstep
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

                        MovieArray[i, j, k].Intensity = Radiation.approximate_solve(MovieArray[i, j, k].Intensity, ji, ki, jf, kf, traj[nstep-1].dl)

                        nstep -= 1
                    end
                    MovieArray[i, j, k].nstep = nstep
                    if nstep != 2
                        do_output = false
                    end
                    lock(progress_lock) do
                        ProgressMeter.next!(p)
                    end
                end
            end

            if do_output
                Image_out = map(x -> x.Intensity, MovieArray[:, :, k]) .* freq^3

                file_name = Printf.format(Printf.Format(output), target_times[k])
                writedlm(file_name, Image_out)
                println("Saving image $(file_name)")

                valid_images[k] = 0
                nopenimgs -= 1
            end
        end
        finish!(p)

        if nopenimgs <= 1
            break
        end
        update_data!(params_slowlight, simulation_data, trat_large, model, all_dumps_path)
    end
end

end
