"""
Brisk-light (reduced temporal fidelity, reduced I/O) rendering: selecting one
representative GRMHD snapshot per lensing band based on the modal emission time
of that band, preserving strong-lensing temporal structure at lower cost than
slow-light.

Phase 1: Temporal interpolation between 2-3 snapshots per band using sliding
window cache, eliminating repeated I/O of identical dumps.

Usage note: Initialize the model with `brisk_light=true, slow_light=false`:
```julia
model = Jipole.Iharm.read_header(..., brisk_light=true, slow_light=false)
```

The `slow_light=false` flag disables the global slow-light path; brisk-light's
per-band interpolation is gated on `model.brisk_light` in Grid.jl.
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

export OfBriskLight, BandWindowState, compute_band_modal_times!, process_brisklight_images!,
    find_band_crossing_time, modal_hdi_kde, clip_ts_to_interval, update_band_window!,
    find_three_dumps_around_time

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
Per-band window state: which 2-3 dumps currently bracket t_target
for temporal interpolation. Tracks the current snapshot window and
allows sliding updates without reloading snapshots unnecessarily.

# Fields
- `indices::Tuple`: Indices of loaded dumps in dump_times (size 2 or 3)
- `snapshots::Vector`: Loaded IharmData snapshots corresponding to indices
- `t_window_start::Float64`: Simulation time of first dump in window
- `t_window_end::Float64`: Simulation time of last dump in window
"""
mutable struct BandWindowState
    indices::Tuple
    snapshots::Vector
    t_window_start::Float64
    t_window_end::Float64
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

    # Bisect over density threshold to find connected interval enclosing mass p
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
For p=0: t_left == t_right == t̄_n, so all pixels map to the modal time.
For p>0: emission times are restricted to the high-density region.
"""
function clip_ts_to_interval(ts_abs::Float64, t_left::Float64, t_right::Float64)::Float64
    return clamp(ts_abs, t_left, t_right)
end


"""
    find_band_crossing_time(traj, nstep, target_band, model)

Walk a single geodesic and return the coordinate time at the step where it
crosses the equatorial plane for the target_band-th time, instead of the
trajectory's final step (which just marks where the integrator stopped,
not where the photon actually met the disk).

# Arguments
- `traj`: Trajectory vector (array of OfTrajS).
- `nstep`: Number of steps in the trajectory.
- `target_band`: The band index to find the crossing for.
- `model`: Iharm model for coordinate transformation.

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
    r, th = Iharm.bl_coord(traj[1].X, model)
    position_in_midplane = th > π/2 ? 1 : 0

    for k in 2:nstep
        r, th = Iharm.bl_coord(traj[k].X, model)
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
    compute_band_modal_times!(midplane_crossings, all_geodesics, nsteps, pixels_x, pixels_y, params_brisklight, model)

Collect emission times per band and compute t̄_n for each.
Must be called before loading snapshots for each frame.

# Arguments
- `midplane_crossings`: Matrix of band indices per pixel.
- `all_geodesics`: Matrix of geodesic trajectories.
- `nsteps`: Matrix of trajectory lengths.
- `pixels_x`, `pixels_y`: Image resolution.
- `params_brisklight`: Brisk-light run state (updated in-place).
- `model`: Iharm model for coordinate transformation.

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

            crossing_time = find_band_crossing_time(all_geodesics[i, j], nsteps[i, j], band_idx, model)
            if crossing_time !== nothing
                push!(band_ts_lists[band_idx + 1], crossing_time)
            end
        end
    end

    # Compute modal time for each band
    for n in 0:params_brisklight.n_bands
        if length(band_ts_lists[n + 1]) > 0
            hdi_result = modal_hdi_kde(band_ts_lists[n + 1], 0.0)
            params_brisklight.modal_times[n + 1] = hdi_result.mode
            @info "Brisk-light: band $n → t̄_$n = $(hdi_result.mode) M  ($(length(band_ts_lists[n + 1])) pixels)"
        else
            params_brisklight.modal_times[n + 1] = 0.0
            @warn "Brisk-light: band $n has no pixels — modal_times[$n] set to 0."
        end
    end

    return band_ts_lists
end


"""
    find_three_dumps_around_time(t_target, dump_times, dump_list)

Find 2 or 3 dumps bracketing t_target.
If t_target is near the boundary, return only the available dumps.

Returned tuple has size 2 (two dumps) or 3 (three dumps):
- (idx_before, idx_after) if insufficient boundary dumps
- (idx_before, idx_best, idx_after) for normal case

# Arguments
- `t_target`: Target simulation time.
- `dump_times`: Vector of simulation times for each dump.
- `dump_list`: Vector of dump file paths.

# Returns
- Tuple of dump indices (2 or 3 elements).
"""
function find_three_dumps_around_time(t_target::Float64, 
                                      dump_times::Vector{Float64}, 
                                      dump_list::Vector{String})::Tuple

    diffs = abs.(dump_times .- t_target)
    best_idx = argmin(diffs)
    
    # Determine which side of best_idx t_target falls
    if t_target < dump_times[best_idx]
        # t_target is between best_idx-1 and best_idx
        idx_before = max(best_idx - 1, 1)
        idx_after = best_idx
    else
        # t_target is between best_idx and best_idx+1
        idx_before = best_idx
        idx_after = min(best_idx + 1, length(dump_list))
    end
    
    # Construction similar to slow-light behavior
    if idx_before == 1 && idx_after == 1
        # Only one dump available: return as pair
        return (1, 1)
    elseif idx_after == idx_before + 1
        # Two consecutive dumps (normal boundary case)
        return (idx_before, idx_after)
    else
        # Three dumps available: include the middle one
        idx_middle = div(idx_before + idx_after, 2)
        return (idx_before, idx_middle, idx_after)
    end
end


"""
    update_band_window!(band_state, t_target, dump_times, dump_list, trat_large, model, dump_cache)

Slide the 2-3 snapshot window forward/backward if t_target falls outside 
[t_window_start, t_window_end]. Reuse snapshots when possible via dump_cache.

Implements the sliding window mechanism: when the target time moves beyond
the current window, load new dumps and release old ones (if not needed by
other bands). This avoids redundant I/O across multiple bands in the same frame.

# Arguments
- `band_state`: BandWindowState to update in-place.
- `t_target`: Target simulation time for this band.
- `dump_times`: Vector of simulation times for each dump.
- `dump_list`: Vector of dump file paths.
- `trat_large`: Electron/ion temperature ratio at high magnetization.
- `model`: Iharm model parameters.
- `dump_cache`: Dict{Int, NamedTuple} of cached (data, valid, refcount) tuples.
"""
function update_band_window!(band_state::BandWindowState,
                             t_target::Float64,
                             dump_times::Vector{Float64},
                             dump_list::Vector{String},
                             trat_large::Float64,
                             model,
                             dump_cache::Dict)

    # Check if t_target is already within the current window
    if band_state.t_window_start <= t_target <= band_state.t_window_end
        return
    end
    
    # Find new window indices
    new_indices = find_three_dumps_around_time(t_target, dump_times, dump_list)
    
    # Load snapshots: reuse from cache if available, load if new
    new_snapshots = Vector(undef, length(new_indices))
    
    for (pos, new_idx) in enumerate(new_indices)
        if haskey(dump_cache, new_idx) && dump_cache[new_idx].valid
            # Reuse from cache (already loaded)
            new_snapshots[pos] = dump_cache[new_idx].data
        else
            # Load new snapshot
            new_snapshots[pos] = Iharm.load_data(dump_list[new_idx], trat_large, model)
            dump_cache[new_idx] = (data = new_snapshots[pos], valid = true)
        end
    end
    
    # Update band window state
    band_state.indices = new_indices
    band_state.snapshots = new_snapshots
    band_state.t_window_start = dump_times[new_indices[1]]
    band_state.t_window_end = dump_times[new_indices[end]]
    
    @info "  Band window updated: t_target = $t_target M → dumps[$(new_indices)] at t = [$(dump_times[new_indices[1]]), $(dump_times[new_indices[end]])] M"
end


"""
    integrate_brisklight_emission!(traj, nsteps, Image, I, J, freq, bhspin, midplane_crossings_ij, data_band_snapshots, model)

Per-pixel radiative transfer with per-band snapshot interpolation.
Same structure as slow-light but the active plasma snapshot is determined
by the band structure rather than photon arrival time.

# Notes
X[1] is not modified: the temporal prescription is implicit in data_band_snapshots[band].
data_band_snapshots[band] is a Vector{IharmData} of 2-3 snapshots, and set_tinterp_ns
will automatically select and interpolate between them.
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
                                        data_band_snapshots::Vector{Vector},
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
    current_band = clamp(midplane_crossings_ij, 0, length(data_band_snapshots) - 1)

    # Track equatorial-plane side to detect crossings during integration.
    # bl_coord returns (r, th) in Boyer-Lindquist coordinates.
    _, th_prev    = Iharm.bl_coord(traj[nsteps].X, model)
    above_prev    = th_prev < π / 2.0

    # data_band_snapshots[band] is already a Vector{IharmData} with 2-3 snapshots
    data_current_vec = data_band_snapshots[current_band + 1]
    
    ji, ki = Radiation.get_jk(traj[nsteps].X, traj[nsteps].Kcon, freq, bhspin, model, data_current_vec)
    Intensity = 0.0

    for nstep in nsteps:-1:2
        for k in 1:NDIM
            Xi[k]    = traj[nstep].X[k]
            Xf[k]    = traj[nstep - 1].X[k]
            Kconi[k] = traj[nstep].Kcon[k]
            Kconf[k] = traj[nstep - 1].Kcon[k]
        end

        if !Radiation.radiating_region(Xf, model, Rh)
            continue
        end

        # Detect equatorial-plane crossing at the endpoint of this step.
        # Walking toward the camera means descending from higher to lower bands.
        _, th_curr = Iharm.bl_coord(Xf, model)
        above_curr = th_curr < π / 2.0

        if above_curr != above_prev
            current_band = max(current_band - 1, 0)
            # Switch to snapshots of the new band
            data_current_vec = data_band_snapshots[current_band + 1]
            above_prev = above_curr
        end
        
        jf, kf = Radiation.get_jk(Xf, Kconf, freq, bhspin, model, data_current_vec)
        
        Intensity = Radiation.approximate_solve(Intensity, ji, ki, jf, kf, traj[nstep - 1].dl)

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

Main brisk-light driver with sliding-window snapshot management.
Generates all frames in [t_obs_start, t_obs_end] while maintaining a per-band
cache of 2-3 GRMHD snapshots, sliding the window as needed and reusing snapshots
across bands to minimize redundant I/O.

# Arguments
- `params_brisklight`: Brisk-light run state.
- `simulation_data`: Vector (unused, kept for API compatibility).
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
    output_dir = "../../data/Images/BriskLight"
    mkpath(output_dir) 
    output_fmt = joinpath(output_dir, "BriskImage.%05d.txt")
    
    Image = zeros(Float64, pixels_x, pixels_y)

    # Compute modal times once, before the t_obs loop,
    # since t̄_n is fixed geometry, not something that depends on t_obs.
    compute_band_modal_times!(midplane_crossings, all_geodesics, nsteps,
                          pixels_x, pixels_y, params_brisklight, model)

    # Initialize per-band sliding windows and global dump cache
    band_windows = [BandWindowState((1, 1), Vector(undef, 2), 0.0, 0.0) 
                    for _ in 0:params_brisklight.n_bands]
    dump_cache = Dict{Int, NamedTuple}()
    
    # data_band_snapshots will hold the active snapshots for each band
    data_band_snapshots = Vector{Vector}(undef, params_brisklight.n_bands + 1)

    # Shift the start/end of the observation window by the per-band
    # light-crossing delay, so that t_obs + t_n always falls inside the range
    # covered by the loaded dumps, for every band, in every frame.
    t_obs = t_obs_start - minimum(params_brisklight.modal_times)

    while t_obs <= t_obs_end - maximum(params_brisklight.modal_times)

        params_brisklight.t_obs = t_obs
        @info "Brisk-light: processing frame at t_obs = $t_obs M"

        # Update sliding window for each band: load new snapshots if t_obs moves
        # the target time outside the current window. Reuse snapshots via cache.
        for n in 0:params_brisklight.n_bands
            t_modal_n = params_brisklight.modal_times[n + 1]
            t_abs_n = t_modal_n + t_obs
            
            # Slide window if necessary, reuse cached snapshots
            update_band_window!(band_windows[n + 1], t_abs_n, dump_times, dump_list,
                               trat_large, model, dump_cache)
            
            # Store reference to the active snapshots for this band
            data_band_snapshots[n + 1] = band_windows[n + 1].snapshots
        end

        # Per-pixel radiative transfer
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
                    data_band_snapshots,
                    model
                )
                ProgressMeter.next!(p)
            end
        end
        finish!(p)

        # Apply ν³ factor and write image to disk
        Image_out = Image .* freq^3
        file_name = Printf.format(Printf.Format(output_fmt), t_obs)
        writedlm(file_name, Image_out)
        println("Brisk-light: image saved → $file_name")

        t_obs += params_brisklight.image_cadence
    end
end


end # module Brisklight
