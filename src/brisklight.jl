"""
Brisk-light (reduced temporal fidelity, reduced I/O) rendering with p generalization.

Brisk light evaluates the same geodesics, redshift factors and source positions as
slow light, and approximates only the *temporal* argument of the source: for each
lensing band n it builds the distribution of emission times, takes the modal
highest-density interval T_{n,p} enclosing probability mass p, and clips the
slow-light emission time to that interval (Eq. V.2 of Rojas-Paternina &
Cardenas-Avendano 2025). p = 0 evaluates the whole band at its modal time; p = 1
bypasses the clipping map and reduces to slow light.

Usage: initialize the model with `brisk_light=true, slow_light=false` and set
`params_brisklight.p` before calling `process_brisklight_images!`.


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
    find_band_crossing_time, collect_band_step_times, modal_hdi_kde, clip_ts_to_interval,
    update_band_window!, find_dump_triplet, brisk_step_times, get_dump_time

using KernelDensity

# Band convention (Jipole):
#   midplane_crossings == 0  ->  shadow / captured photons
#   midplane_crossings == 1  ->  direct image     (n = 0 in AART / in the paper)
#   midplane_crossings == 2  ->  first indirect   (n = 1)
#   midplane_crossings == 3  ->  second indirect  (n = 2)

"""
Brisk-light run state.

# Fields
- `n_bands::Int`: Number of lensing bands
- `modal_times::Vector{Float64}`: t̄_n per band, in geometric time (negative)
- `hdi_intervals::Vector{Tuple}`: T_{n,p} = [t_left, t_right] per band, geometric time
- `band_time_ranges::Vector{Tuple}`: [t_min, t_max] of sampled step times per band
- `p::Float64`: Probability mass parameter in [0, 1]
- `image_cadence::Float64`: Time step between frames
- `t_obs::Float64`: Current observation time
"""
mutable struct OfBriskLight
    n_bands::Int
    modal_times::Vector{Float64}
    hdi_intervals::Vector{Tuple}
    band_time_ranges::Vector{Tuple}
    p::Float64
    image_cadence::Float64
    t_obs::Float64
end

"""
Per-band window state: which dump triplet currently brackets the band's target
time. `t_window_start` / `t_window_end` are the absolute simulation times of the
first and last snapshot, and are handed to the integrator as the availability
clamp (tA, tB). A triplet is required by `set_tinterp_ns`; see
`find_dump_triplet`.
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

KDE-based modal HDI, valid for any p in [0, 1].

p = 0 returns only the mode. p = 1 returns `(-Inf, Inf)`: the clipping map is
bypassed entirely, since `clamp(t, -Inf, Inf) == t`. Returning the trimmed
quantiles instead would still clip the tails and prevent convergence to slow
light; the paper notes that the exact slow-light result is obtained by bypassing
the clipping map and using the full ray-traced time map directly.
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
    p == 1.0 && return (mode = t_modal, interval = (-Inf, Inf),        mass = 1.0)

    # Bisect over the density threshold until the connected component around the
    # mode encloses probability mass p.
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

Clip a single emission time to the HDI (Eq. V.2). Kept for diagnostics; the
integrator uses `brisk_step_times`, which clips a *pair* rigidly.
"""
function clip_ts_to_interval(ts_abs::Float64, t_left::Float64, t_right::Float64)::Float64
    return clamp(ts_abs, t_left, t_right)
end

"""
    find_band_crossing_time(traj, nstep, target_band, model)

Walk a geodesic and return the coordinate time at the equatorial-plane crossing
that defines `target_band`. Kept as a diagnostic: the KDE is now built from
step-level times (see `collect_band_step_times`), not from crossing times.
"""
function find_band_crossing_time(traj::Vector, nstep::Int, target_band::Int, model)
    if target_band == 0
        return traj[nstep].X[1]
    end

    crossing_count = 0
    r, th = Iharm.bl_coord(traj[1].X, model)
    position_in_midplane = th > pi/2 ? 1 : 0

    for k in 2:nstep
        r, th = Iharm.bl_coord(traj[k].X, model)
        if (position_in_midplane == 1) && (th <= pi/2)
            position_in_midplane = 0
            crossing_count += 1
        elseif (position_in_midplane == 0) && (th > pi/2)
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
    collect_band_step_times(midplane_crossings, all_geodesics, nsteps, pixels_x,
                            pixels_y, n_bands, model, Rh; pixel_stride)

Sample the emission-time distribution at the *integration-step* level, band by band.

The band walk mirrors `integrate_brisklight_emission!` exactly (start at
`midplane_crossings[i,j]`, decrement at each equatorial crossing while walking
back toward the camera), so the HDI of band n describes precisely the set of
steps that will later be clipped to it.

This is what makes the p = 1 limit meaningful. Building the KDE from one crossing
time per pixel while applying the clipping map at every step compares two
different domains: the interval would be tens of M wide while the step times span
hundreds of M, so the map could never become the identity.

`pixel_stride` subsamples the screen; the KDE needs a converged distribution, not
every step. Check that t̄_n is stable between stride 1 and stride 4.
"""
function collect_band_step_times(midplane_crossings::Matrix{Int},
                                 all_geodesics,
                                 nsteps::Matrix{Int},
                                 pixels_x::Int,
                                 pixels_y::Int,
                                 n_bands::Int,
                                 model,
                                 Rh::Float64;
                                 pixel_stride::Int = 4)

    band_ts = [Float64[] for _ in 0:n_bands]

    for i in 1:pixel_stride:pixels_x, j in 1:pixel_stride:pixels_y
        n = nsteps[i, j]
        n < 2 && continue
        traj = all_geodesics[i, j]

        current_band = clamp(midplane_crossings[i, j], 0, n_bands)
        _, th_prev   = Iharm.bl_coord(traj[n].X, model)
        above_prev   = th_prev < pi / 2.0

        for nstep in n:-1:2
            Xf = traj[nstep - 1].X
            Radiation.radiating_region(Xf, model, Rh) || continue

            _, th_curr = Iharm.bl_coord(Xf, model)
            above_curr = th_curr < pi / 2.0
            if above_curr != above_prev
                current_band = max(current_band - 1, 0)
                above_prev   = above_curr
            end

            push!(band_ts[current_band + 1], Xf[1])
        end
    end
    return band_ts
end

"""
    compute_band_modal_times!(midplane_crossings, all_geodesics, nsteps, pixels_x,
                              pixels_y, params_brisklight, model, p; pixel_stride)

Compute t̄_n, the HDI T_{n,p}, and the sampled time range for every band.
Must be called before rendering. Updates `params_brisklight` in place.
"""
function compute_band_modal_times!(midplane_crossings::Matrix{Int},
                                   all_geodesics,
                                   nsteps::Matrix{Int},
                                   pixels_x::Int,
                                   pixels_y::Int,
                                   params_brisklight::OfBriskLight,
                                   model,
                                   p::Float64;
                                   pixel_stride::Int = 4)

    Rh = 1.0 + sqrt(1.0 - model.a * model.a)

    band_ts_lists = collect_band_step_times(midplane_crossings, all_geodesics, nsteps,
                                            pixels_x, pixels_y,
                                            params_brisklight.n_bands, model, Rh;
                                            pixel_stride = pixel_stride)

    params_brisklight.p = p
    params_brisklight.hdi_intervals    = Vector{Tuple}(undef, params_brisklight.n_bands + 1)
    params_brisklight.band_time_ranges = Vector{Tuple}(undef, params_brisklight.n_bands + 1)

    for n in 0:params_brisklight.n_bands
        ts = band_ts_lists[n + 1]
        if length(ts) >= 2
            hdi_result = modal_hdi_kde(ts, p)
            params_brisklight.modal_times[n + 1]      = hdi_result.mode
            params_brisklight.hdi_intervals[n + 1]    = hdi_result.interval
            params_brisklight.band_time_ranges[n + 1] = (minimum(ts), maximum(ts))
            @info "Brisk-light: band $n -> p=$p -> t_modal_$n = $(hdi_result.mode) M, " *
                  "HDI = [$(hdi_result.interval[1]), $(hdi_result.interval[2])] M " *
                  "($(length(ts)) step samples)"
        else
            params_brisklight.modal_times[n + 1]      = 0.0
            params_brisklight.hdi_intervals[n + 1]    = (0.0, 0.0)
            params_brisklight.band_time_ranges[n + 1] = (0.0, 0.0)
            @warn "Brisk-light: band $n has too few step samples - skipped"
        end
    end

    return band_ts_lists
end

"""
    find_dump_triplet(t_target, dump_times)

Return three consecutive dump indices `(i, i+1, i+2)` bracketing `t_target`.

Three, not two: the generic `set_tinterp_ns` method in `iharm.jl` (line 724) does

    nA, nB = X[1] < data[2].t ? (1, 2) : (2, 3)

so it indexes `data[3]` whenever the query time sits in the upper half of the
window. Handing it a 2-element vector is a latent `BoundsError`. The valid
domain of a triplet is `[dump_times[i], dump_times[i+2]]`, i.e. two dump
spacings of temporal support per band per frame.

Consequence: when the retained width W_n(p) exceeds 2*dT, the availability clamp
in `brisk_step_times` dominates and p stops being resolvable. Lifting that
requires `set_tinterp_ns` to bracket-search over N ordered snapshots.
"""
function find_dump_triplet(t_target::Float64, dump_times::Vector{Float64})::Tuple
    N = length(dump_times)
    N >= 3 || error("find_dump_triplet: need at least three dumps, got $N.")
    i = clamp(searchsortedlast(dump_times, t_target) - 1, 1, N - 2)
    return (i, i + 1, i + 2)
end

"""
    update_band_window!(band_state, t_target, dump_times, dump_list, trat_large,
                        model, dump_cache)

Slide the snapshot triplet for a band when `t_target` leaves the current
bracket.
Snapshots are reused across bands and frames through `dump_cache`.
"""
function update_band_window!(band_state::BandWindowState,
                             t_target::Float64,
                             dump_times::Vector{Float64},
                             dump_list::Vector{String},
                             trat_large::Float64,
                             model,
                             dump_cache::Dict)

    if band_state.t_window_start <= t_target <= band_state.t_window_end
        return
    end

    new_indices = find_dump_triplet(t_target, dump_times)
    new_snapshots = Vector(undef, length(new_indices))

    for (pos, new_idx) in enumerate(new_indices)
        if haskey(dump_cache, new_idx) && dump_cache[new_idx].valid
            new_snapshots[pos] = dump_cache[new_idx].data
        else
            new_snapshots[pos] = Iharm.load_data(dump_list[new_idx], trat_large, model)
            dump_cache[new_idx] = (data = new_snapshots[pos], valid = true)
        end
    end

    band_state.indices        = new_indices
    band_state.snapshots      = new_snapshots
    band_state.t_window_start = dump_times[new_indices[1]]
    band_state.t_window_end   = dump_times[new_indices[end]]

    @info "  Band window updated: t_target = $t_target M -> dumps[$(new_indices)] " *
          "at t = [$(band_state.t_window_start), $(band_state.t_window_end)] M"
end

"""
    brisk_step_times(t_i_geo, t_f_geo, t_obs, t_left, t_right, tA, tB)

Map a geodesic step pair from geometric time to absolute simulation time,
applying the brisk-light clipping map and then the snapshot-availability clamp.

Both maps translate the pair *rigidly*, preserving `t_f - t_i`. Clipping each
endpoint independently would collapse the pair separation to zero whenever both
endpoints fall on the same side of the interval - which, before this fix,
happened for the large majority of steps. That separation is what
`approximate_solve` integrates over.

The availability clamp mirrors `slowlight.jl`: when a step falls outside the
loaded triplet, both endpoints shift together so that the interpolation weight
computed in `set_tinterp_ns` stays inside [0, 1] instead of extrapolating the
plasma. `tA` / `tB` are the first and last snapshot times of the triplet.

With p = 1 the HDI is `(-Inf, Inf)`, the first block is the identity, and what
remains is exactly slow light's mapping.
"""
@inline function brisk_step_times(t_i_geo::Float64, t_f_geo::Float64,
                                  t_obs::Float64,
                                  t_left::Float64, t_right::Float64,
                                  tA::Float64, tB::Float64)
    ti = t_i_geo + t_obs
    tf = t_f_geo + t_obs
    dt = tf - ti

    # Brisk-light clipping map (Eq. V.2), as a rigid translation.
    tL = t_left  + t_obs
    tR = t_right + t_obs
    if ti < tL
        ti = tL; tf = ti + dt
    elseif tf > tR
        tf = tR; ti = tf - dt
    end

    # Snapshot availability clamp, matching slowlight.jl.
    if ti < tA
        ti = tA; tf = ti + dt
    elseif tf > tB
        tf = tB; ti = tf - dt
    end
    return ti, tf
end

"""
    integrate_brisklight_emission!(...)

Per-pixel radiative transfer. The band index selects both the active snapshot
pair and the HDI used to clip the step times; it is decremented at each
equatorial crossing while walking back toward the camera.
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
                                        hdi_intervals::Vector{Tuple},
                                        band_window_times::Vector{Tuple},
                                        t_obs::Float64,
                                        model)

    NDIM = 4
    Xi    = MVector{4,Float64}(undef)
    Kconi = MVector{4,Float64}(undef)
    Xf    = MVector{4,Float64}(undef)
    Kconf = MVector{4,Float64}(undef)

    Xi_abs = MVector{4,Float64}(undef)
    Xf_abs = MVector{4,Float64}(undef)

    Rh = 1.0 + sqrt(1.0 - bhspin * bhspin)

    for k in 1:NDIM
        Xi[k]    = traj[nsteps].X[k]
        Kconi[k] = traj[nsteps].Kcon[k]
    end

    current_band = clamp(midplane_crossings_ij, 0, length(data_band_snapshots) - 1)
    t_left, t_right  = hdi_intervals[current_band + 1]
    tA_band, tB_band = band_window_times[current_band + 1]

    _, th_prev = Iharm.bl_coord(traj[nsteps].X, model)
    above_prev = th_prev < pi / 2.0

    data_current_vec = data_band_snapshots[current_band + 1]

    # Seed point: same mapping applied to a degenerate pair.
    ti_abs, _ = brisk_step_times(Xi[1], Xi[1], t_obs, t_left, t_right, tA_band, tB_band)
    for k in 1:NDIM
        Xi_abs[k] = Xi[k]
    end
    Xi_abs[1] = ti_abs

    ji, ki = Radiation.get_jk(Xi_abs, traj[nsteps].Kcon, freq, bhspin, model, data_current_vec)
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

        _, th_curr = Iharm.bl_coord(Xf, model)
        above_curr = th_curr < pi / 2.0

        if above_curr != above_prev
            current_band = max(current_band - 1, 0)
            data_current_vec = data_band_snapshots[current_band + 1]
            t_left, t_right  = hdi_intervals[current_band + 1]
            tA_band, tB_band = band_window_times[current_band + 1]
            above_prev = above_curr
        end

        _, tf_abs = brisk_step_times(Xi[1], Xf[1], t_obs, t_left, t_right, tA_band, tB_band)
        for k in 1:NDIM
            Xf_abs[k] = Xf[k]
        end
        Xf_abs[1] = tf_abs

        jf, kf = Radiation.get_jk(Xf_abs, Kconf, freq, bhspin, model, data_current_vec)

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
    process_brisklight_images!(params_brisklight, simulation_data, all_geodesics,
        nsteps, midplane_crossings, model, pixels_x, pixels_y, freq, trat_large,
        all_dumps_path, dump_list, dump_times, tA, tB, tf; pixel_stride)

Main brisk-light driver. `tA`, `tB`, `tf` bound the available dump range.
Output goes to a p-specific directory so that different p values never overwrite
each other.
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
    trat_large::Float64,
    all_dumps_path::String,
    dump_list::Vector{String},
    dump_times::Vector{Float64},
    tA::Float64,
    tB::Float64,
    tf::Float64;
    pixel_stride::Int = 4
)
    output_dir = "../../data/Images/BriskLight_p$(params_brisklight.p)"
    mkpath(output_dir)
    output_fmt = joinpath(output_dir, "BriskImage.%05d.txt")

    Image = zeros(Float64, pixels_x, pixels_y)

    compute_band_modal_times!(midplane_crossings, all_geodesics, nsteps,
                              pixels_x, pixels_y, params_brisklight, model,
                              params_brisklight.p; pixel_stride = pixel_stride)

    band_windows = [BandWindowState((1, 2, 3), Vector(undef, 3), 0.0, 0.0)
                    for _ in 0:params_brisklight.n_bands]
    dump_cache = Dict{Int, NamedTuple}()
    data_band_snapshots = Vector{Vector}(undef, params_brisklight.n_bands + 1)
    band_window_times   = Vector{Tuple}(undef, params_brisklight.n_bands + 1)

    # Band times are negative geometric delays, so the *earliest* source time a
    # frame needs is set by the most negative one. Starting at tA - t_band_latest
    # would push the earliest bands before the first dump.
    t_band_latest   = maximum(params_brisklight.modal_times)
    t_band_earliest = minimum(params_brisklight.modal_times)

    last_img_target = tA - t_band_earliest

    nimg = 0

    while true
        if (last_img_target + t_band_latest >= tB)
            break
        end

        if !(last_img_target + t_band_earliest < tf - model.rmax_geo)
            @info "Brisk-light: skipping invalid frame at t_obs = $last_img_target M"
            last_img_target += params_brisklight.image_cadence
            continue
        end

        params_brisklight.t_obs = last_img_target
        @info "Brisk-light: processing frame at t_obs = $last_img_target M (p = $(params_brisklight.p))"

        for n in 0:params_brisklight.n_bands
            t_lo_geo, t_hi_geo = params_brisklight.hdi_intervals[n + 1]
            r_lo, r_hi         = params_brisklight.band_time_ranges[n + 1]

            # Anchor the pair on the centre of the retained support, clipped to
            # the times the band actually samples (the p = 1 HDI is infinite).
            t_lo = max(t_lo_geo, r_lo)
            t_hi = min(t_hi_geo, r_hi)
            t_anchor = 0.5 * (t_lo + t_hi) + last_img_target

            update_band_window!(band_windows[n + 1], t_anchor, dump_times, dump_list,
                                trat_large, model, dump_cache)

            data_band_snapshots[n + 1] = band_windows[n + 1].snapshots
            band_window_times[n + 1]   = (band_windows[n + 1].t_window_start,
                                          band_windows[n + 1].t_window_end)
        end

        fill!(Image, 0.0)

        p_bar = Progress(
            pixels_x * pixels_y;
            desc      = "Brisk-light t_obs = $last_img_target M",
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
                    params_brisklight.hdi_intervals,
                    band_window_times,
                    last_img_target,
                    model
                )
                ProgressMeter.next!(p_bar)
            end
        end
        finish!(p_bar)

        Image_out = Image .* freq^3
        frame_index = round(Int, last_img_target)

        file_name = Printf.format(Printf.Format(output_fmt), frame_index)
        writedlm(file_name, Image_out)
        println("Brisk-light: image saved -> $file_name")

        nimg += 1

        # Evict snapshots no longer referenced by any band window.
        keep = Set{Int}()
        for n in 0:params_brisklight.n_bands
            union!(keep, band_windows[n + 1].indices)
        end
        for k in collect(keys(dump_cache))
            k in keep || delete!(dump_cache, k)
        end

        last_img_target += params_brisklight.image_cadence
    end

    @info "Brisk-light: rendering complete (p = $(params_brisklight.p), $nimg frames generated)"
end

end # module Brisklight
