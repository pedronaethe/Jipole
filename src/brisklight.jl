"""
Brisk-light (reduced temporal fidelity, reduced I/O) rendering: selecting one
representative GRMHD snapshot per lensing band based on the modal emission time
of that band, preserving strong-lensing temporal structure at lower cost than
slow-light.

Usage note: Initialize the model with `brisk_light=true, slow_light=false`:
```julia
model = Jipole.Iharm.read_header(..., brisk_light=true, slow_light=false)
```

The `slow_light=false` flag disables temporal interpolation (which is unnecessary
for brisk-light since each band already uses a single modal snapshot).
"""
module Brisklight

using HDF5
using Printf
using DelimitedFiles
using ProgressMeter
using StaticArrays
using Statistics
using ..Constants
using ..Radiation
using ..Iharm

export OfBriskLight, compute_band_modal_times!, process_brisklight_images!,
    find_band_crossing_time, modal_hdi_kde, clip_ts_to_interval

# Band convention (Jipole):
#   midplane_crossings == 0  →  shadow 
#   midplane_crossings == 1  →  direct image       (n=0 in AART)
#   midplane_crossings == 2  →  first indirect     (n=1 in AART)
#   midplane_crossings == 3  →  second indirect    (n=2 in AART)

using KernelDensity

"""
Global state for brisk-light, analogous to OfSlowLight
"""
mutable struct OfBriskLight
    n_bands::Int
    modal_times::Vector{Float64}   # t̄_n per band
    image_cadence::Float64
    t_obs::Float64
end

"""
    get_dump_time(dump_idx, all_dumps_path)

Read coordinate time from a dump file.

# Arguments
- `dump_idx`: Index of the dump in the sequence.
- `all_dumps_path`: `Printf`-style format string for the dump sequence.

# Returns
- The simulation time of the dump.
"""
function get_dump_time(dump_idx::Int, all_dumps_path::String)::Float64
    dump_path = Printf.format(Printf.Format(all_dumps_path), dump_idx)
    t::Float64 = 0.0
    h5open(dump_path, "r") do file
        t = read(file, "t")
    end
    return t
end

"""
    modal_hdi_kde(ts_band, p; trim_quantiles, gridsize)

KDE-based modal HDI — works for any p in [0,1].
p=0 returns only the mode; p=1 returns the full trimmed support.

# Arguments
- `ts_band`: Vector of emission times for a single band.
- `p`: Probability mass to enclose in the HDI, in [0, 1].
- `trim_quantiles`: Tuple of (low, high) quantiles for trimming outliers.
- `gridsize`: Resolution of the KDE grid.

# Returns
- Named tuple: (mode, interval, mass)
"""
function modal_hdi_kde(ts_band::Vector{Float64}, p::Float64;
                       trim_quantiles::Tuple{Float64, Float64} = (0.005, 0.995),
                       gridsize::Int = 4096)

    0.0 <= p <= 1.0 || error("modal_hdi_kde: p must be in [0, 1], got p = $p")

    x = filter(isfinite, ts_band)
    length(x) >= 2 || error("modal_hdi_kde: need at least 2 finite samples.")

    q_low, q_high = trim_quantiles
    lo = quantile(x, q_low)
    hi = quantile(x, q_high)
    lo < hi || error("modal_hdi_kde: invalid trim bounds (lo=$lo, hi=$hi).")

    x_trim = filter(v -> lo <= v <= hi, x)
    length(x_trim) >= 2 || error("modal_hdi_kde: too few samples after trimming.")

    k        = kde(x_trim; boundary = (lo, hi), npoints = gridsize)
    density  = k.density
    x_grid   = collect(k.x)
    dx       = step(k.x)

    mode_idx = argmax(density)
    t_modal  = x_grid[mode_idx]

    p == 0.0 && return (mode = t_modal, interval = (t_modal, t_modal), mass = 0.0)
    p == 1.0 && return (mode = t_modal, interval = (lo, hi),           mass = 1.0)

    # bisect over density threshold to find connected interval enclosing mass p
    function modal_component(threshold)
        above = density .>= threshold
        l, r  = mode_idx, mode_idx
        while l > 1               && above[l - 1]; l -= 1; end
        while r < length(density) && above[r + 1]; r += 1; end
        return l, r
    end

    low_thresh  = 0.0
    high_thresh = density[mode_idx]

    for _ in 1:60
        mid  = 0.5 * (low_thresh + high_thresh)
        l, r = modal_component(mid)
        if sum(density[l:r]) * dx >= p
            low_thresh = mid
        else
            high_thresh = mid
        end
    end

    l_f, r_f = modal_component(low_thresh)
    t_left   = x_grid[l_f]
    t_right  = x_grid[r_f]
    mass     = sum(density[l_f:r_f]) * dx

    return (mode = t_modal, interval = (t_left, t_right), mass = mass)
end


"""
    clip_ts_to_interval(ts_abs, t_left, t_right)

Clip emission time to HDI [t_left, t_right].
p=0: t_left == t_right == t̄_n, so all pixels map to the modal time.
"""
function clip_ts_to_interval(ts_abs::Float64, t_left::Float64, t_right::Float64)::Float64
    return clamp(ts_abs, t_left, t_right)
end


"""
    find_band_crossing_time(traj, nstep, target_band)

Walk a single geodesic and return the coordinate time at the step where it
crosses the equatorial plane for the target_band-th time, instead of the
trajectory's final step (which just marks where the integrator stopped,
not where the photon actually met the disk).

# Arguments
- `traj`: Trajectory vector (array of OfTrajS).
- `nstep`: Number of steps in the trajectory.
- `target_band`: The band index to find the crossing for.

# Returns
- Coordinate time X[1] at the band crossing, or nothing if not reached.

# Notes
Band 0 (shadow/captured photons) never crosses the midplane by definition,
so there is no crossing to search for — falls back to the deepest point reached.
Captured photons still radiate: they traverse emitting plasma before the horizon.
"""
function find_band_crossing_time(traj::Vector, nstep::Int, target_band::Int, model)
    if target_band == 0
        return traj[nstep].X[1]
    end

    crossing_count = 0
    r, th = Iharm.bl_coord(traj[1].X, model)      # ← CON model
    position_in_midplane = th > π/2 ? 1 : 0

    for k in 2:nstep
        r, th = Iharm.bl_coord(traj[k].X, model)  # ← CON model
        if (position_in_midplane == 1) && (th <= π/2)
            position_in_midplane = 0
            crossing_count += 1
        elseif (position_in_midplane == 0) && (th > π/2)
            position_in_midplane = 1
            crossing_count += 1
        end
        if crossing_count == target_band
            return traj[k].X[1]
        end
    end
    return nothing
end

"""
    compute_band_modal_times!(midplane_crossings, all_geodesics, nsteps, pixels_x, pixels_y, params_brisklight)

Collect emission times per band and compute t̄_n for each.
Must be called before loading snapshots for each frame.

# Arguments
- `midplane_crossings`: Matrix of band indices per pixel.
- `all_geodesics`: Matrix of geodesic trajectories.
- `nsteps`: Matrix of trajectory lengths.
- `pixels_x`, `pixels_y`: Image resolution.
- `params_brisklight`: Brisk-light run state (updated in-place).

# Notes
t_obs is no longer a parameter here. The geodesics are fixed geometry traced
once, so t_n is purely geometric and only needs to be computed a single time,
not on every frame of the t_obs loop.
"""
function compute_band_modal_times!(midplane_crossings::Matrix{Int},
                                   all_geodesics,
                                   nsteps::Matrix{Int},
                                   pixels_x::Int,
                                   pixels_y::Int,
                                   params_brisklight::OfBriskLight,
                                   model)

    band_ts_lists = [Float64[] for _ in 0:params_brisklight.n_bands]

    for i in 1:pixels_x
        for j in 1:pixels_y

            band_idx = midplane_crossings[i, j]
            (band_idx < 0 || band_idx > params_brisklight.n_bands) && continue

            n = nsteps[i, j]
            n < 2 && continue

            # Use the actual disk-crossing time for this band instead
            # of the trajectory's last stored step.
            t_cross = find_band_crossing_time(all_geodesics[i, j], n, band_idx, model)
            t_cross === nothing && continue

            t_abs = t_cross
            push!(band_ts_lists[band_idx + 1], t_abs)
        end
    end

    # modal time per band
    for n in 0:params_brisklight.n_bands
        ts_n = band_ts_lists[n + 1]

        if length(ts_n) >= 2
            result = modal_hdi_kde(ts_n, 0.0)
            params_brisklight.modal_times[n + 1] = result.mode
            @info "Brisk-light: band $n → t̄_$n = $(result.mode) M  ($(length(ts_n)) pixels)"
        elseif length(ts_n) == 1
            params_brisklight.modal_times[n + 1] = ts_n[1]
            @warn "Brisk-light: band $n has only 1 pixel — using its time as modal time."
        else
            params_brisklight.modal_times[n + 1] = 0.0
            @warn "Brisk-light: band $n has no pixels — modal_times[$n] set to 0."
        end
    end

    return band_ts_lists
end


"""
    integrate_brisklight_emission!(traj, nsteps, Image, I, J, freq, bhspin, midplane_crossings_ij, simulation_data)

Per-pixel radiative transfer — same structure as integrate_emission! in radiation.jl
but switches the active plasma snapshot when the geodesic crosses the equatorial plane.

# Notes
X[1] is never modified: the temporal prescription is implicit in simulation_data[band].
Captured photons radiate: pixels with midplane_crossings == 0 should not be zeroed —
captured photons traverse emitting plasma before the horizon.
"""
function integrate_brisklight_emission!(traj,
                                        nsteps::Int,
                                        Image::Matrix{Float64},
                                        I::Int,
                                        J::Int,
                                        freq::Float64,
                                        bhspin::Float64,
                                        midplane_crossings_ij::Int,
                                        simulation_data::Vector,
                                        model)

    NDIM = 4
    
    Xi    = MVector{4,Float64}(undef)
    Kconi = MVector{4,Float64}(undef)
    Xf    = MVector{4,Float64}(undef)
    Kconf = MVector{4,Float64}(undef)
    
    Rh    = 1.0 + sqrt(1.0 - bhspin * bhspin)

    # Starting point: step farthest from camera (closest to disk)
    for k in 1:NDIM
        Xi[k]    = traj[nsteps].X[k]
        Kconi[k] = traj[nsteps].Kcon[k]
    end

    # Band at the start of the walk (deepest point of the geodesic).
    # midplane_crossings_ij directly gives the Jipole band index.
    current_band = clamp(midplane_crossings_ij, 0, length(simulation_data) - 1)

    # Track equatorial-plane side to detect crossings during integration.
    # bl_coord returns (r, th) in Boyer-Lindquist coordinates.
    _, th_prev    = Iharm.bl_coord(traj[nsteps].X,model)
    above_prev    = th_prev < π / 2.0

    data_current_vec = Vector{Iharm.IharmData}(undef, 1)
    data_current_vec[1] = simulation_data[current_band + 1]
    
    ji, ki = Radiation.get_jk(traj[nsteps].X, traj[nsteps].Kcon, freq, bhspin, model, data_current_vec)
    Intensity = 0.0

    for nstep in nsteps:-1:2
        for k in 1:NDIM
            Xi[k]    = traj[nstep].X[k]
            Xf[k]    = traj[nstep - 1].X[k]
            Kconi[k] = traj[nstep].Kcon[k]
            Kconf[k] = traj[nstep - 1].Kcon[k]
        end

        if !Radiation.radiating_region(Xf,model, Rh)
            continue
        end

        # Detect equatorial-plane crossing at the endpoint of this step.
        # Walking toward the camera means descending from higher to lower bands.
        _, th_curr = Iharm.bl_coord(Xf,model)
        above_curr = th_curr < π / 2.0

        if above_curr != above_prev
            current_band            = max(current_band - 1, 0)
            data_current_vec[1]     = simulation_data[current_band + 1]
            above_prev              = above_curr
        end
        
        jf, kf = Radiation.get_jk(Xf, Kconf, freq, bhspin, model, data_current_vec)
        
        Intensity  = Radiation.approximate_solve(Intensity, ji, ki, jf, kf, traj[nstep - 1].dl)

        if isnan(Intensity) || isinf(Intensity)
            @error "Invalid intensity in brisk-light at pixel ($I, $J), step $nstep"
            error("Intensity is $Intensity")
        end

        ji = jf
        ki = kf
    end

    Image[I, J] = Intensity
end


"""
    process_brisklight_images!(params_brisklight, simulation_data, all_geodesics, nsteps, midplane_crossings, model, pixels_x, pixels_y, freq, t_obs_start, t_obs_end, trat_large, all_dumps_path, dump_list, dump_times)

Main brisk-light driver — generates all frames in [t_obs_start, t_obs_end].

# Arguments
- `params_brisklight`: Brisk-light run state.
- `simulation_data`: Vector of loaded GRMHD snapshots (one per band).
- `all_geodesics`: Matrix of pre-traced geodesic trajectories.
- `nsteps`: Matrix of trajectory lengths.
- `midplane_crossings`: Matrix of band indices.
- `model`: Iharm model parameters.
- `pixels_x`, `pixels_y`: Image resolution.
- `freq`: Frequency in cgs units.
- `t_obs_start`, `t_obs_end`: Observation window bounds.
- `trat_large`: Electron/ion temperature ratio at high magnetization.
- `all_dumps_path`: `Printf`-style format string for the dump sequence.
- `dump_list`: Vector of dump file paths.
- `dump_times`: Vector of simulation times for each dump.
"""
function process_brisklight_images!(
    params_brisklight::OfBriskLight,
    simulation_data::Vector,
    all_geodesics,
    nsteps::Matrix{Int},
    midplane_crossings::Matrix{Int},
    model,
    pixels_x::Int,
    pixels_y::Int,
    freq::Float64,
    t_obs_start::Float64,
    t_obs_end::Float64,
    trat_large::Float64,
    all_dumps_path::String,
    dump_list::Vector{String},
    dump_times::Vector{Float64}
)
    output_dir = "../../data/Images/BriskLight" ##Poner en variables de entrada 
    mkpath(output_dir) 
    output_fmt = joinpath(output_dir, "BriskImage.%05d.txt")
    
    Image      = zeros(Float64, pixels_x, pixels_y)

    # Compute modal times once, before the t_obs loop,
    # instead of once per frame — t̄_n is fixed geometry, not something that
    # depends on t_obs.
    compute_band_modal_times!(midplane_crossings, all_geodesics, nsteps,
                          pixels_x, pixels_y, params_brisklight, model)

    # Shift the start/end of the observation window by the per-band
    # light-crossing delay, so that t_obs + t_n always falls inside the range
    # covered by the loaded dumps, for every band, in every frame.
    t_obs = t_obs_start - minimum(params_brisklight.modal_times)

    while t_obs <= t_obs_end - maximum(params_brisklight.modal_times)

        params_brisklight.t_obs = t_obs
        @info "Brisk-light: processing frame at t_obs = $t_obs M"

        #load one snapshot per band (closest dump to t_n)
        for n in 0:params_brisklight.n_bands
            t_modal_n   = params_brisklight.modal_times[n + 1]
            # t_obs is added exactly once — t_modal_n already is the
            # absolute-time-independent geometric delay for this band.
            t_abs_n     =  t_modal_n + t_obs
            dump_path_n = find_dump_path_for_time(t_abs_n, dump_times, dump_list)
            simulation_data[n + 1] = Iharm.load_data(dump_path_n, trat_large, model)
            @info "  Band $n: snapshot loaded at t̄_$n = $t_modal_n M  ← $dump_path_n"
        end

        # per-pixel radiative transfer
        fill!(Image, 0.0)

        p = Progress(
            pixels_x * pixels_y;
            desc      = "Brisk-light t_obs = $t_obs M",
            showspeed = true,
            barlen    = 30
        )

        Threads.@threads for i in 1:pixels_x
            for j in 1:pixels_y
                integrate_brisklight_emission!(
                    all_geodesics[i, j],
                    nsteps[i, j],
                    Image, i, j,
                    freq, model.a,
                    midplane_crossings[i, j],
                    simulation_data,
                    model
                )
                ProgressMeter.next!(p)
            end
        end
        finish!(p)

        #apply ν³ factor and write image to disk
        Image_out = Image .* freq^3
        file_name = Printf.format(Printf.Format(output_fmt), t_obs)
        writedlm(file_name, Image_out)
        println("Brisk-light: image saved → $file_name")

        t_obs += params_brisklight.image_cadence
    end
end


"""
    find_dump_path_for_time(t_target, dump_times, dump_list)

Nearest-neighbour search over dump_list/dump_times.
Brisk-light needs non-consecutive dumps so we can't advance sequentially.

# Arguments
- `t_target`: Target simulation time.
- `dump_times`: Vector of simulation times for each dump.
- `dump_list`: Vector of dump file paths.

# Returns
- Path to the closest dump.
"""
function find_dump_path_for_time(t_target::Float64, dump_times::Vector{Float64}, dump_list::Vector{String})::String
    diffs    = abs.(dump_times .- t_target)
    best_idx = argmin(diffs)
    
    @info "  find_dump_path_for_time: t_target = $t_target M → dump[$best_idx] at t = $(dump_times[best_idx]) M"
    
    return dump_list[best_idx]
end

end # module Brisklight
