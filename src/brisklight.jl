"""
Brisk-light (reduced temporal fidelity, reduced I/O) rendering with p generalization.

Brisk light evaluates the same geodesics, redshift factors and source positions as
slow light, and approximates only the *temporal* argument of the source: for each
lensing band n it builds the distribution of emission times, takes the modal
highest-density interval T_{n,p} enclosing probability mass p, and clips the
slow-light emission time to that interval (Eq. V.2 of Rojas-Paternina &
Cardenas-Avendano). p = 0 evaluates the whole band at its modal time; p = 1
bypasses the clipping map and reduces to slow light *provided* every band's
retained support fits inside its loaded snapshot window (see
`availability_clamp_report`).

Usage
    model  = Iharm.read_header(...; brisk_light=true, slow_light=false)
    params = OfBriskLight(n_bands, p, ImageCadence)
    process_brisklight_images!(params, all_geodesics, nsteps, midplane_crossings,
                               model, pixels_x, pixels_y, freq, trat_large,
                               dump_list, dump_times, tA, tB, tf;
                               t_obs_start = tA - tgeof)


"""
module Brisklight

using HDF5
using Printf
using DelimitedFiles
using ProgressMeter
using StaticArrays
using Statistics
using KernelDensity
using ..Constants
using ..Radiation
using ..Iharm

export OfBriskLight, BandWindowState,
    compute_band_modal_times!, process_brisklight_images!,
    collect_band_step_times, band_time_extrema,
    precompute_step_info,
    modal_hdi_kde,
    update_band_window!, find_dump_window, brisk_step_times, get_dump_time,
    retained_support, band_anchor, availability_clamp_report

# 
# Band convention
# Two distinct quantities share the word "band" and must not be confused.
#
#   * `midplane_crossings[i,j]` is a PER-PIXEL count returned by
#     `Geodesics.get_pixel`: the total number of times that ray crosses
#     theta = pi/2 anywhere along its trajectory.
#         0 -> captured / shadow, 1 -> direct image, 2 -> first indirect, ...
#     i.e. it is offset by one relative to the AART index n.
#
#   * `current_band` is a RUNNING counter used to label geodesic SEGMENTS. It is
#     seeded at `midplane_crossings[i,j]` and decremented at each crossing while
#     walking forward along the ray (from the far end toward the camera). The
#     segment bounded by crossings n and n-1 therefore carries `current_band == n`,
#     which does coincide with the AART band index.
#
# One consequence has no AART counterpart: the segment traversed BEFORE the
# first crossing (the inbound approach) keeps the seed value
# `midplane_crossings[i,j]`. Set `approach_bin = true` to route those steps to a
# dedicated bin at index `n_bands + 1` instead of polluting band
# `midplane_crossings[i,j]`. Likewise `shadow_bin = true` routes every step of a
# captured ray (`midplane_crossings == 0`) to index `n_bands + 2` instead of
# mixing it into band 0.
#
# Array layout used throughout: index `n + 1` holds band n for n = 0..n_bands,
# index `n_bands + 2` holds the approach bin, index `n_bands + 3` the shadow bin.
# `nbins(n_bands) == n_bands + 3`.

@inline nbins(n_bands::Int) = n_bands + 3
@inline approach_bin_index(n_bands::Int) = n_bands + 2
@inline shadow_bin_index(n_bands::Int)   = n_bands + 3

"""
Brisk-light run state.

# Fields
- `n_bands::Int`: highest AART band index tracked
- `modal_times::Vector{Float64}`: t̄_n per bin, geometric time (negative); `NaN` if invalid
- `hdi_intervals::Vector{NTuple{2,Float64}}`: T_{n,p} per bin, geometric time
- `band_time_ranges::Vector{NTuple{2,Float64}}`: [t_min, t_max] of sampled times per bin
- `band_valid::Vector{Bool}`: false for bins with too few samples; excluded from all reductions
- `p::Float64`: retained probability mass, in [0, 1]
- `image_cadence::Float64`: observer-time step between frames
- `t_obs::Float64`: observer time of the frame currently being rendered
"""
mutable struct OfBriskLight
    n_bands::Int
    modal_times::Vector{Float64}
    hdi_intervals::Vector{NTuple{2,Float64}}
    band_time_ranges::Vector{NTuple{2,Float64}}
    band_valid::Vector{Bool}
    p::Float64
    image_cadence::Float64
    t_obs::Float64
end

"""
    OfBriskLight(n_bands, p, image_cadence)

Convenience constructor; all per-band arrays are sized and zeroed.
"""
function OfBriskLight(n_bands::Int, p::Real, image_cadence::Real)
    nb = nbins(n_bands)
    return OfBriskLight(n_bands,
                        fill(NaN, nb),
                        fill((NaN, NaN), nb),
                        fill((NaN, NaN), nb),
                        fill(false, nb),
                        Float64(p),
                        Float64(image_cadence),
                        0.0)
end

"""
Per-band snapshot window: which contiguous run of dumps currently brackets the
band's anchor time. `t_window_start` / `t_window_end` are the absolute simulation
times of the first and last snapshot and are handed to the integrator as the
availability clamp (tA, tB).

Parametric in the snapshot type so that the inner transfer loop stays type
stable; the concrete type is inferred from a probe load in
`process_brisklight_images!`.
"""
mutable struct BandWindowState{D}
    indices::UnitRange{Int}
    snapshots::Vector{D}
    t_window_start::Float64
    t_window_end::Float64
end

BandWindowState{D}() where {D} =
    BandWindowState{D}(1:0, Vector{D}(), Inf, -Inf)

# Dump metadata

"""
    get_dump_time(dump_idx, all_dumps_path)

Read the coordinate time `t` from a dump without loading fluid primitives.
"""
function get_dump_time(dump_idx::Int, all_dumps_path::String)::Float64
    dump_path = Printf.format(Printf.Format(all_dumps_path), dump_idx)
    t::Float64 = 0.0
    h5open(dump_path, "r") do file
        t = read(file, "t")
    end
    return t
end

# Modal highest-density interval

"""
    modal_hdi_kde(ts, p; trim_quantiles, gridsize, bandwidth)

KDE-based modal HDI, valid for any p in [0, 1]. Returns
`(mode, interval, mass, bandwidth, x_grid, density)`. `x_grid`/`density` are the
full KDE curve on the trimmed domain (post-trim, pre-clip), handy for plotting
the same density the HDI was computed from without recomputing it.

`trim_quantiles` defaults to the central fraction q = 0.995 used in the paper,
i.e. quantiles (0.0025, 0.9975). Trimming defines the smooth density only; no
sample is removed from the image.

p = 0 returns only the mode. p = 1 returns `(-Inf, Inf)`: the clipping map is
bypassed entirely, since `clamp(t, -Inf, Inf) == t`. Returning the trimmed
quantiles instead would still clip the tails and prevent convergence to slow
light.
"""
function modal_hdi_kde(ts::AbstractVector{Float64}, p::Real;
                       trim_quantiles::Tuple{Float64,Float64} = (0.0025, 0.9975),
                       gridsize::Int = 4096,
                       bandwidth::Union{Nothing,Float64} = nothing)

    p = Float64(p)
    0.0 <= p <= 1.0 || error("modal_hdi_kde: p must be in [0, 1], got p = $p")

    x = filter(isfinite, ts)
    length(x) >= 2 || error("modal_hdi_kde: need at least 2 finite samples.")

    q_low, q_high = trim_quantiles
    lo = quantile(x, q_low)
    hi = quantile(x, q_high)
    lo < hi || error("modal_hdi_kde: degenerate trim bounds (lo = $lo, hi = $hi).")

    x_trim = filter(v -> lo <= v <= hi, x)
    length(x_trim) >= 2 || error("modal_hdi_kde: too few samples after trimming.")

    h_used = bandwidth === nothing ? _default_bandwidth(x_trim) : bandwidth
    k = kde(x_trim; boundary = (lo, hi), npoints = gridsize, bandwidth = h_used)

    density = k.density
    x_grid  = collect(k.x)
    dx      = step(k.x)
    ng      = length(density)

    mode_idx = argmax(density)
    t_modal  = x_grid[mode_idx]

    p == 0.0 && return (mode = t_modal, interval = (t_modal, t_modal),
                        mass = 0.0, bandwidth = h_used, x_grid = x_grid, density = density)
    p == 1.0 && return (mode = t_modal, interval = (-Inf, Inf),
                        mass = 1.0, bandwidth = h_used, x_grid = x_grid, density = density)

    # Cumulative mass, so the enclosed mass of a component is O(1) instead of a
    # fresh sum (and no BitVector is allocated per bisection step).
    cmass = cumsum(density) .* dx

    component_mass(l::Int, r::Int) = cmass[r] - (l > 1 ? cmass[l - 1] : 0.0)

    # Connected component of {density >= threshold} containing the global mode.
    function modal_component(threshold::Float64)
        l = mode_idx
        r = mode_idx
        while l > 1  && density[l - 1] >= threshold; l -= 1; end
        while r < ng && density[r + 1] >= threshold; r += 1; end
        return l, r
    end

    low_thresh  = 0.0
    high_thresh = density[mode_idx]

    for _ in 1:60
        mid  = 0.5 * (low_thresh + high_thresh)
        l, r = modal_component(mid)
        if component_mass(l, r) >= p
            low_thresh = mid
        else
            high_thresh = mid
        end
    end

    l_f, r_f = modal_component(low_thresh)
    return (mode      = t_modal,
            interval  = (x_grid[l_f], x_grid[r_f]),
            mass      = component_mass(l_f, r_f),
            bandwidth = h_used,
            x_grid    = x_grid,
            density   = density)
end

"""
Default kernel bandwidth. Uses KernelDensity's own rule when available so that
the unscanned default matches the library, and falls back to Silverman's rule
otherwise.
"""
function _default_bandwidth(x::AbstractVector{Float64})
    try
        return KernelDensity.default_bandwidth(x)
    catch
        n = length(x)
        s = std(x)
        iqr = quantile(x, 0.75) - quantile(x, 0.25)
        a = iqr > 0 ? min(s, iqr / 1.349) : s
        return 0.9 * a * n^(-0.2)
    end
end

# Delay sampling

"""
Resolve the array bin for a running band counter.

`seed` is `midplane_crossings[i,j]`; `band` is the running counter. Steps taken
before the first crossing still carry `band == seed` and are routed to the
approach bin when `approach_bin` is set. Rays with `seed == 0` never cross and
are routed to the shadow bin when `shadow_bin` is set.
"""
@inline function _bin_index(band::Int, seed::Int, n_bands::Int,
                            approach_bin::Bool, shadow_bin::Bool)
    if shadow_bin && seed == 0
        return shadow_bin_index(n_bands)
    end
    if approach_bin && band == seed && seed > 0
        return approach_bin_index(n_bands)
    end
    return clamp(band, 0, n_bands) + 1
end

"""
    collect_band_step_times(midplane_crossings, all_geodesics, nsteps, pixels_x,
                            pixels_y, n_bands, model, Rh; kwargs...)

Sample the emission-time distribution at the *integration-step* level, bin by
bin. Returns `times`, a `Vector{Vector{Float64}}` of length
`nbins(n_bands)`.

The walk mirrors `integrate_brisklight_emission!` exactly, so the HDI of a bin
describes precisely the set of steps that will later be clipped to it. Crossings
are detected *before* the radiating-region filter is applied, so the counter
tracks `trace_geodesic` even when a ray crosses the midplane outside
`rmin_geo < r < rmax_geo`.

Step-level rather than crossing-level sampling is what makes p = 1 meaningful:
building the KDE from one crossing time per pixel while clipping at every step
compares two different domains, and the map could never become the identity. The
price is that the resulting widths are set jointly by the Kerr delay structure
and by the geometric extent of the emitting region, so they are NOT directly
comparable to the equatorial widths quoted in the paper.

# Keywords
- `pixel_stride`: screen subsampling for the KDE (the mode converges fast).
- `approach_bin`, `shadow_bin`: see the band-convention note at the top.
"""
function collect_band_step_times(midplane_crossings::Matrix{Int},
                                 all_geodesics,
                                 nsteps::Matrix{Int},
                                 pixels_x::Int,
                                 pixels_y::Int,
                                 n_bands::Int,
                                 model,
                                 Rh::Float64;
                                 pixel_stride::Int = 4,
                                 approach_bin::Bool = true,
                                 shadow_bin::Bool = true)

    nb      = nbins(n_bands)
    band_ts = [Float64[] for _ in 1:nb]

    for i in 1:pixel_stride:pixels_x, j in 1:pixel_stride:pixels_y
        n = nsteps[i, j]
        n < 2 && continue
        traj = all_geodesics[i, j]

        seed         = midplane_crossings[i, j]
        current_band = seed
        _, th_prev   = Iharm.bl_coord(traj[n].X, model)
        above_prev   = th_prev < pi / 2.0

        for nstep in n:-1:2
            Xf = traj[nstep - 1].X

            # Crossing bookkeeping first, so that crossings outside the emitting
            # shell still advance the counter (matches trace_geodesic).
            _, th_curr = Iharm.bl_coord(Xf, model)
            above_curr = th_curr < pi / 2.0
            if above_curr != above_prev
                current_band = max(current_band - 1, 0)
                above_prev   = above_curr
            end

            Radiation.radiating_region(Xf, model, Rh) || continue

            b = _bin_index(current_band, seed, n_bands, approach_bin, shadow_bin)
            push!(band_ts[b], Xf[1])
        end
    end
    return band_ts
end

"""
    band_time_extrema(...)

Per-bin `(t_min, t_max)` of the sampled step times, computed without building
the sample vectors. Cheap enough to run at `pixel_stride = 1`, which matters
because the extrema converge far more slowly than the mode and they are what set
the frame schedule.
"""
function band_time_extrema(midplane_crossings::Matrix{Int},
                           all_geodesics,
                           nsteps::Matrix{Int},
                           pixels_x::Int,
                           pixels_y::Int,
                           n_bands::Int,
                           model,
                           Rh::Float64;
                           pixel_stride::Int = 1,
                           approach_bin::Bool = true,
                           shadow_bin::Bool = true)

    nb   = nbins(n_bands)
    tmin = fill(Inf, nb)
    tmax = fill(-Inf, nb)

    for i in 1:pixel_stride:pixels_x, j in 1:pixel_stride:pixels_y
        n = nsteps[i, j]
        n < 2 && continue
        traj = all_geodesics[i, j]

        seed         = midplane_crossings[i, j]
        current_band = seed
        _, th_prev   = Iharm.bl_coord(traj[n].X, model)
        above_prev   = th_prev < pi / 2.0

        for nstep in n:-1:2
            Xf = traj[nstep - 1].X

            _, th_curr = Iharm.bl_coord(Xf, model)
            above_curr = th_curr < pi / 2.0
            if above_curr != above_prev
                current_band = max(current_band - 1, 0)
                above_prev   = above_curr
            end

            Radiation.radiating_region(Xf, model, Rh) || continue

            b = _bin_index(current_band, seed, n_bands, approach_bin, shadow_bin)
            t = Xf[1]
            t < tmin[b] && (tmin[b] = t)
            t > tmax[b] && (tmax[b] = t)
        end
    end
    return [(tmin[b], tmax[b]) for b in 1:nb]
end

# Band statistics

"""
    compute_band_modal_times!(params, band_ts, ranges, p)

Fill `modal_times`, `hdi_intervals`, `band_time_ranges` and `band_valid` from a
sample that has already been collected. Use this form when sweeping p, so that
the geodesics are walked once instead of once per p.
"""
function compute_band_modal_times!(params_brisklight::OfBriskLight,
                                   band_ts::Vector{Vector{Float64}},
                                   ranges::Vector{NTuple{2,Float64}},
                                   p::Real;
                                   min_samples::Int = 32,
                                   kde_kwargs...)

    nb = nbins(params_brisklight.n_bands)
    length(band_ts) == nb || error("compute_band_modal_times!: expected $nb bins.")

    params_brisklight.p = Float64(p)

    for b in 1:nb
        ts = band_ts[b]
        if length(ts) >= min_samples && (maximum(ts) > minimum(ts))
            res = modal_hdi_kde(ts, p; kde_kwargs...)
            params_brisklight.modal_times[b]      = res.mode
            params_brisklight.hdi_intervals[b]    = res.interval
            params_brisklight.band_time_ranges[b] = ranges[b]
            params_brisklight.band_valid[b]       = true
        else
            params_brisklight.modal_times[b]      = NaN
            params_brisklight.hdi_intervals[b]    = (NaN, NaN)
            params_brisklight.band_time_ranges[b] = (NaN, NaN)
            params_brisklight.band_valid[b]       = false
            length(ts) > 0 && @warn "Brisk-light: bin $b has only $(length(ts)) samples - marked invalid"
        end
    end
    return params_brisklight
end

"""
    compute_band_modal_times!(midplane_crossings, all_geodesics, nsteps, pixels_x,
                              pixels_y, params, model, p; kwargs...)

Collect the sample and fill the band statistics in one call. Returns
`(band_ts, ranges)` so the caller can cache them across a p sweep.
"""
function compute_band_modal_times!(midplane_crossings::Matrix{Int},
                                   all_geodesics,
                                   nsteps::Matrix{Int},
                                   pixels_x::Int,
                                   pixels_y::Int,
                                   params_brisklight::OfBriskLight,
                                   model,
                                   p::Real;
                                   pixel_stride::Int = 4,
                                   extrema_stride::Int = 1,
                                   approach_bin::Bool = true,
                                   shadow_bin::Bool = true,
                                   min_samples::Int = 32,
                                   verbose::Bool = true,
                                   kde_kwargs...)

    Rh = 1.0 + sqrt(1.0 - model.a * model.a)
    nb = params_brisklight.n_bands

    band_ts = collect_band_step_times(midplane_crossings, all_geodesics, nsteps,
                                      pixels_x, pixels_y, nb, model, Rh;
                                      pixel_stride = pixel_stride,
                                      approach_bin = approach_bin,
                                      shadow_bin   = shadow_bin)

    ranges = band_time_extrema(midplane_crossings, all_geodesics, nsteps,
                               pixels_x, pixels_y, nb, model, Rh;
                               pixel_stride = extrema_stride,
                               approach_bin = approach_bin,
                               shadow_bin   = shadow_bin)

    compute_band_modal_times!(params_brisklight, band_ts, ranges, p;
                              min_samples = min_samples,
                              kde_kwargs...)

    if verbose
        for b in 1:nbins(nb)
            params_brisklight.band_valid[b] || continue
            lo, hi = params_brisklight.hdi_intervals[b]
            rl, rh = params_brisklight.band_time_ranges[b]
            @info @sprintf("Brisk-light bin %s (p=%.3f): t_modal = %10.3f M  HDI = [%.3f, %.3f]  range = [%.3f, %.3f]  (%d samples)",
                           _bin_name(b, nb), params_brisklight.p,
                           params_brisklight.modal_times[b], lo, hi, rl, rh,
                           length(band_ts[b]))
        end
    end

    return band_ts, ranges
end

function _bin_name(b::Int, n_bands::Int)
    b == approach_bin_index(n_bands) && return "approach"
    b == shadow_bin_index(n_bands)   && return "shadow"
    return "n=$(b - 1)"
end

"""
    retained_support(params) -> (t_lo, t_hi)

Earliest and latest *geometric* source time a frame actually needs at the
current p, reduced over valid bins:

    t_lo = min_n max(HDI_lo_n, range_lo_n)
    t_hi = max_n min(HDI_hi_n, range_hi_n)

At p = 0 this collapses onto the modal times; at p = 1 it is the full sampled
support. Both limits are correct, which is why the frame schedule keys off this
rather than off the modes.

Note that the retained support shrinks with p, so a low-p run can render a
*longer* observer window from the same dumps than slow light can. Report it.
"""
function retained_support(params_brisklight::OfBriskLight)
    t_lo =  Inf
    t_hi = -Inf
    for b in 1:nbins(params_brisklight.n_bands)
        params_brisklight.band_valid[b] || continue
        hlo, hhi = params_brisklight.hdi_intervals[b]
        rlo, rhi = params_brisklight.band_time_ranges[b]
        lo = max(hlo, rlo)
        hi = min(hhi, rhi)
        isfinite(lo) && lo < t_lo && (t_lo = lo)
        isfinite(hi) && hi > t_hi && (t_hi = hi)
    end
    (isfinite(t_lo) && isfinite(t_hi)) ||
        error("retained_support: no valid band. Run compute_band_modal_times! first.")
    return t_lo, t_hi
end

"""
    band_anchor(params, b) -> Float64

Geometric time on which band `b`'s snapshot window is centred. Defaults to the
mode, clamped into the retained support. Delay distributions are strongly
asymmetric about their mode, so anchoring on the midpoint of the interval places
the window off-centre relative to where the mass sits and maximises the clamped
fraction.
"""
function band_anchor(params_brisklight::OfBriskLight, b::Int; anchor::Symbol = :mode)
    hlo, hhi = params_brisklight.hdi_intervals[b]
    rlo, rhi = params_brisklight.band_time_ranges[b]
    lo = max(hlo, rlo)
    hi = min(hhi, rhi)
    anchor === :midpoint && return 0.5 * (lo + hi)
    anchor === :mode     && return clamp(params_brisklight.modal_times[b], lo, hi)
    error("band_anchor: anchor must be :mode or :midpoint.")
end

# Snapshot windows

"""
    find_dump_window(t_target, dump_times; n_window = 3)

Return a contiguous run of `n_window` dump indices bracketing `t_target`, chosen
so that the target sits as close as possible to the centre of the run.

Three is the minimum: the generic `set_tinterp_ns` in `iharm.jl` does

    nA, nB = X[1] < data[2].t ? (1, 2) : (2, 3)

so it indexes `data[3]` whenever the query sits in the upper half of the window,
and a 2-element vector is a latent `BoundsError`. The valid domain of a run is
`[dump_times[first], dump_times[last]]`, i.e. `(n_window - 1)` dump spacings of
temporal support per band per frame.

`n_window > 3` requires the generalised `set_tinterp_ns` that bracket-searches
over N ordered snapshots. Without that patch, keep `n_window = 3`.
"""
function find_dump_window(t_target::Float64, dump_times::Vector{Float64};
                          n_window::Int = 3)::UnitRange{Int}
    N = length(dump_times)
    n_window >= 3 || error("find_dump_window: n_window must be at least 3.")
    N >= n_window || error("find_dump_window: need at least $n_window dumps, got $N.")

    k = clamp(searchsortedlast(dump_times, t_target), 1, N)

    # Centre the run on t_target: the middle element should be the nearest dump.
    half = (n_window - 1) ÷ 2
    i    = clamp(k - half, 1, N - n_window + 1)

    # One-step refinement so the target is not systematically in the upper half.
    if i + 1 <= N - n_window + 1
        c_now  = abs(dump_times[i + half]     - t_target)
        c_next = abs(dump_times[i + half + 1] - t_target)
        c_next < c_now && (i += 1)
    end
    return i:(i + n_window - 1)
end

"""
    update_band_window!(band_state, t_target, dump_times, dump_list, trat_large,
                        model, dump_cache; n_window)

Slide a band's snapshot run when `t_target` leaves the current bracket.
Snapshots are reused across bands and frames through `dump_cache`, so bands
whose anchors fall within the same spacing cost a single load.
"""
function update_band_window!(band_state::BandWindowState{D},
                             t_target::Float64,
                             dump_times::Vector{Float64},
                             dump_list::Vector{String},
                             trat_large::Float64,
                             model,
                             dump_cache::Dict{Int,D};
                             n_window::Int = 3,
                             verbose::Bool = false) where {D}

    if band_state.t_window_start <= t_target <= band_state.t_window_end &&
       length(band_state.snapshots) == n_window
        return 0
    end

    new_indices   = find_dump_window(t_target, dump_times; n_window = n_window)
    new_snapshots = Vector{D}(undef, length(new_indices))
    n_loaded      = 0

    for (pos, idx) in enumerate(new_indices)
        if haskey(dump_cache, idx)
            new_snapshots[pos] = dump_cache[idx]
        else
            new_snapshots[pos] = Iharm.load_data(dump_list[idx], trat_large, model)
            dump_cache[idx]    = new_snapshots[pos]
            n_loaded          += 1
        end
    end

    band_state.indices        = new_indices
    band_state.snapshots      = new_snapshots
    band_state.t_window_start = dump_times[first(new_indices)]
    band_state.t_window_end   = dump_times[last(new_indices)]

    verbose && @info @sprintf("  window -> dumps[%d:%d]  t = [%.3f, %.3f] M  (%d loads)",
                              first(new_indices), last(new_indices),
                              band_state.t_window_start, band_state.t_window_end, n_loaded)
    return n_loaded
end

# The time map

"""
    brisk_step_times(t_i_geo, t_f_geo, t_obs, t_left, t_right, tA, tB)

Map a geodesic step pair from geometric to absolute simulation time, applying
the brisk-light clipping map and then the snapshot-availability clamp.

Each endpoint is clipped independently, which is exactly Eq. V.2. The transfer
integral is taken over the affine step length `dl` (see
`Radiation.approximate_solve`), not over `t_f - t_i`, so collapsing a pair onto a
single time costs nothing: it simply evaluates the plasma at the same instant at
two different spatial points, which is precisely what p = 0 prescribes.

The availability clamp keeps the interpolation weight computed in
`set_tinterp_ns` inside [0, 1] instead of extrapolating the plasma. Unlike slow
light, which defers an out-of-window step to a later round, brisk light must
evaluate it now, so a clamped step is a locally frozen plasma. Monitor the
clamped fraction with `availability_clamp_report`; a brisk run whose clamped
fraction is not negligible is not testing brisk light.

With p = 1 the HDI is `(-Inf, Inf)`, the first clamp is the identity, and what
remains is slow light's mapping restricted to the loaded window.
"""
@inline function brisk_step_times(t_i_geo::Float64, t_f_geo::Float64,
                                  t_obs::Float64,
                                  t_left::Float64, t_right::Float64,
                                  tA::Float64, tB::Float64)
    tL = t_left  + t_obs
    tR = t_right + t_obs
    ti = clamp(clamp(t_i_geo + t_obs, tL, tR), tA, tB)
    tf = clamp(clamp(t_f_geo + t_obs, tL, tR), tA, tB)
    return ti, tf
end

# Per-pixel transfer
"""
    precompute_step_info(midplane_crossings, all_geodesics, nsteps, pixels_x,
                         pixels_y, n_bands, model, Rh; approach_bin, shadow_bin)

Precompute, once per pixel, the two per-step quantities the render loop used to
recompute on every call: whether a step lies inside the emitting shell
(`Radiation.radiating_region`), and which bin (band / approach / shadow) it
belongs to. Both are pure functions of the traced geodesic's SPATIAL coordinates
`(r, theta)`, which never change; only the time coordinate `X[1]` varies with
`t_obs`, and `p` only changes the HDI, not the band walk.

This exists because `integrate_brisklight_emission!` is called once per pixel,
per frame, per p in a sweep. Before this cache, every one of those calls redid
the crossing walk from scratch with `Iharm.bl_coord`, including over the long
non-radiating outbound leg `r: rmax_geo .. ro` (equatorial-crossing detection was
moved ahead of the radiating-region filter to fix the band-closure bug, F10, so
it can no longer be skipped by a `continue`). Across a full p sweep that is easily
billions of redundant coordinate transforms. Precomputing removes it entirely
from the hot loop, leaving a single array lookup per step.

Returns `(radiating, bin)`, each `Matrix{Vector}` indexed `[i,j][nstep]` exactly
the way the render loop is: entry `nstep` describes the segment ending at
`traj[nstep - 1]`. `radiating` is `Vector{Bool}`, `bin` is `Vector{Int8}` holding
the 1-based index into `hdi_intervals` / `data_bins`.
"""
function precompute_step_info(midplane_crossings::Matrix{Int},
                              all_geodesics,
                              nsteps::Matrix{Int},
                              pixels_x::Int,
                              pixels_y::Int,
                              n_bands::Int,
                              model,
                              Rh::Float64;
                              approach_bin::Bool = true,
                              shadow_bin::Bool = true)

    radiating = Matrix{Vector{Bool}}(undef, pixels_x, pixels_y)
    bin       = Matrix{Vector{Int8}}(undef, pixels_x, pixels_y)

    Threads.@threads :greedy for i in 1:pixels_x
        for j in 1:pixels_y
            n = nsteps[i, j]
            n_alloc = max(n, 1)
            rad = Vector{Bool}(undef, n_alloc)
            bn  = Vector{Int8}(undef, n_alloc)
            seed = midplane_crossings[i, j]

            if n == 1
                # Degenerate geodesic (camera point only). The render loop's
                # seed line reads index `nsteps == 1` before the main loop ever
                # runs, so it must be defined even though it is never really
                # exercised as a transfer step.
                bn[1]  = Int8(clamp(seed, 0, n_bands) + 1)
                rad[1] = false
            elseif n >= 2
                traj = all_geodesics[i, j]
                current_band = seed
                _, th_prev   = Iharm.bl_coord(traj[n].X, model)
                above_prev   = th_prev < pi / 2.0

                for nstep in n:-1:2
                    Xf = traj[nstep - 1].X

                    _, th_curr = Iharm.bl_coord(Xf, model)
                    above_curr = th_curr < pi / 2.0
                    if above_curr != above_prev
                        current_band = max(current_band - 1, 0)
                        above_prev   = above_curr
                    end

                    rad[nstep] = Radiation.radiating_region(Xf, model, Rh)
                    bn[nstep]  = Int8(_bin_index(current_band, seed, n_bands,
                                                 approach_bin, shadow_bin))
                end
            end

            radiating[i, j] = rad
            bin[i, j]       = bn
        end
    end
    return radiating, bin
end

"""
    integrate_brisklight_emission!(...)

Per-pixel radiative transfer. `step_bin[nstep]` selects both the active snapshot
run and the HDI used to clip the step times; `step_radiating[nstep]` gates
whether the step contributes. Both come from `precompute_step_info` and encode
the equatorial-crossing walk (band decremented at each crossing while walking
forward toward the camera, detected before the radiating-region filter, so the
band counter matches `collect_band_step_times` and `trace_geodesic`) - none of
that bookkeeping happens in this function any more; it is pure array lookups,
which is what keeps this loop cheap across a p sweep.

One detail differs from a naive port of `Radiation.integrate_emission!`: when
the bin changes, the leading emissivity `(ji, ki)` is re-evaluated against the
new snapshot run. Otherwise a single transfer step would average `j` and
`alpha` taken from two different epochs, at the equatorial plane, where the
emission is brightest.
"""
function integrate_brisklight_emission!(traj,
                                        nsteps::Int,
                                        Image::Matrix{Float64},
                                        I::Int,
                                        J::Int,
                                        freq::Float64,
                                        bhspin::Float64,
                                        step_radiating::Vector{Bool},
                                        step_bin::Vector{Int8},
                                        data_bins::Vector{Vector{D}},
                                        hdi_intervals::Vector{NTuple{2,Float64}},
                                        bin_window_times::Vector{NTuple{2,Float64}},
                                        t_obs::Float64,
                                        model,
                                        bad_pixels::Threads.Atomic{Int}) where {D}

    NDIM = 4
    Xi    = MVector{4,Float64}(undef)
    Kconi = MVector{4,Float64}(undef)
    Xf    = MVector{4,Float64}(undef)
    Kconf = MVector{4,Float64}(undef)
    Xabs  = MVector{4,Float64}(undef)

    b = Int(step_bin[nsteps])
    t_left, t_right  = hdi_intervals[b]
    tA_band, tB_band = bin_window_times[b]
    data_current     = data_bins[b]

    for k in 1:NDIM
        Xi[k]    = traj[nsteps].X[k]
        Kconi[k] = traj[nsteps].Kcon[k]
    end

    # Seed point: the same map applied to a degenerate pair.
    ti_abs, _ = brisk_step_times(Xi[1], Xi[1], t_obs, t_left, t_right, tA_band, tB_band)
    for k in 1:NDIM
        Xabs[k] = Xi[k]
    end
    Xabs[1] = ti_abs

    ji, ki = Radiation.get_jk(Xabs, Kconi, freq, bhspin, model, data_current)
    Intensity = 0.0

    # Set when the active bin changes; cleared only once the leading emissivity
    # has actually been re-evaluated, so a crossing that falls in a non-radiating
    # gap does not leave (ji, ki) stranded on the previous bin's snapshots.
    pending_reeval = false

    for nstep in nsteps:-1:2
        for k in 1:NDIM
            Xi[k]    = traj[nstep].X[k]
            Xf[k]    = traj[nstep - 1].X[k]
            Kconi[k] = traj[nstep].Kcon[k]
            Kconf[k] = traj[nstep - 1].Kcon[k]
        end

        b_new = Int(step_bin[nstep])
        if b_new != b
            b = b_new
            t_left, t_right  = hdi_intervals[b]
            tA_band, tB_band = bin_window_times[b]
            data_current     = data_bins[b]
            pending_reeval   = true
        end

        step_radiating[nstep] || continue

        # Re-evaluate the leading endpoint against the new bin, so that a single
        # transfer step never mixes two epochs.
        if pending_reeval
            ti_new, _ = brisk_step_times(Xi[1], Xi[1], t_obs, t_left, t_right, tA_band, tB_band)
            for k in 1:NDIM
                Xabs[k] = Xi[k]
            end
            Xabs[1] = ti_new
            ji, ki = Radiation.get_jk(Xabs, Kconi, freq, bhspin, model, data_current)
            pending_reeval = false
        end

        _, tf_abs = brisk_step_times(Xi[1], Xf[1], t_obs, t_left, t_right, tA_band, tB_band)
        for k in 1:NDIM
            Xabs[k] = Xf[k]
        end
        Xabs[1] = tf_abs

        jf, kf = Radiation.get_jk(Xabs, Kconf, freq, bhspin, model, data_current)

        Intensity = Radiation.approximate_solve(Intensity, ji, ki, jf, kf, traj[nstep - 1].dl)

        if !isfinite(Intensity)
            Threads.atomic_add!(bad_pixels, 1)
            Image[I, J] = NaN
            return
        end

        ji = jf
        ki = kf
    end

    Image[I, J] = Intensity
    return
end

# ---------------------------------------------------------------------------
# Availability-clamp accounting (used internally by process_brisklight_images!)
# ---------------------------------------------------------------------------

"""
    availability_clamp_report(params, band_ts, bin_window_times, t_obs)
        -> (frac_total, frac_per_bin)

Fraction of sampled steps whose clipped time falls outside the loaded snapshot
run and is therefore evaluated against a frozen plasma. Called once per frame
by `process_brisklight_images!` to decide whether to warn that the loaded
snapshot window is too narrow for the current p (frozen-plasma contamination
above ~1% means results should not be compared to slow light).
"""
function availability_clamp_report(params_brisklight::OfBriskLight,
                                   band_ts::Vector{Vector{Float64}},
                                   bin_window_times::Vector{NTuple{2,Float64}},
                                   t_obs::Float64)
    nb        = nbins(params_brisklight.n_bands)
    per_bin   = fill(NaN, nb)
    n_total   = 0
    n_clamped = 0

    for b in 1:nb
        params_brisklight.band_valid[b] || continue
        ts = band_ts[b]
        isempty(ts) && continue
        tL, tR = params_brisklight.hdi_intervals[b]
        tA, tB = bin_window_times[b]
        c = 0
        for t in ts
            ta = clamp(t + t_obs, tL + t_obs, tR + t_obs)
            (ta < tA || ta > tB) && (c += 1)
        end
        per_bin[b] = c / length(ts)
        n_total   += length(ts)
        n_clamped += c
    end
    return (n_total == 0 ? NaN : n_clamped / n_total), per_bin
end

# Driver

"""
    process_brisklight_images!(params, all_geodesics, nsteps, midplane_crossings,
        model, pixels_x, pixels_y, freq, trat_large, dump_list, dump_times,
        tA, tB, tf; kwargs...)

Main brisk-light driver.

Frame scheduling keys off the retained support (`retained_support`), not off the
band modes: the earliest source time a frame needs is `t_lo`, the latest is
`t_hi`, and the schedule runs over

    t_obs in [tA - t_lo,  min(tB, tf - tf_buffer) - t_hi]

which mirrors slow light's `tA - tgeof` ... `tf - rmax_geo - tgeoi`. Pass
`t_obs_start` (and optionally `t_obs_stop`) to place a brisk run on exactly the
same observer-time grid as a slow-light run, or to keep the grid fixed across a
p sweep. Note that the default schedule is p-dependent, because a low-p run
needs less delay support and can therefore cover a longer observer window from
the same dumps; that is a result worth reporting, but it makes frames from
different p values non-comparable unless the grid is pinned.

A manifest `frames.csv` recording `(index, t_obs, file, clamped_fraction,
n_loads)` is written next to the images. Pair runs by `t_obs`, never by list
position: slow light silently drops frames whose integration never completes,
so the two file lists can desynchronise halfway through.

Two things are independent of `t_obs` and of `p`, so they are computed once and
reused for every frame of this call: the delay sample (`band_ts`) and the
per-step radiating flag / bin index (`step_radiating`/`step_bin`, see
`precompute_step_info`). The latter is what keeps the render loop free of
`Iharm.bl_coord` calls, which otherwise dominate the CPU cost of a p sweep -
every pixel's long non-radiating outbound leg (`r: rmax_geo .. ro`) used to be
re-walked from scratch on every single frame of every single `p`. Pass all four
(`band_ts`, `ranges`, `step_radiating`, `step_bin`) back in on subsequent calls
of a sweep - the return value includes them for exactly this purpose - to pay
for both only once.

# Keywords
- `t_obs_start`, `t_obs_stop`: pin the observer-time grid.
- `output_dir`, `output_prefix`, `frame_index_mode` (`:time` or `:sequential`).
- `pixel_stride`, `extrema_stride`: KDE subsampling and extrema subsampling.
- `n_window`: snapshots per band (3 unless `set_tinterp_ns` is generalised).
- `anchor`: `:mode` (default) or `:midpoint`.
- `approach_bin`, `shadow_bin`: separate the inbound segment and captured rays.
  Also shrinks the resident-snapshot ceiling from `3*(n_bands+3)` back to
  `3*(n_bands+1)` when disabled, at the cost of folding those two auxiliary
  populations back into bands 0 and `n_bands`.
- `band_ts`, `ranges`: pass a cached delay sample to skip re-walking the
  geodesics during a p sweep.
- `step_radiating`, `step_bin`: pass a cached per-step cache (from a prior call,
  or from `precompute_step_info` directly) to skip re-deriving it; this is the
  expensive one to recompute, not the delay sample.
- `tf_buffer`: safety margin subtracted from `tf` (defaults to `model.rmax_geo`,
  matching slow light).
- `clamp_warn`: warn when the clamped fraction of a frame exceeds this.
"""
function process_brisklight_images!(
    params_brisklight::OfBriskLight,
    all_geodesics,
    nsteps::Matrix{Int},
    midplane_crossings::Matrix{Int},
    model,
    pixels_x::Int,
    pixels_y::Int,
    freq::Float64,
    trat_large::Float64,
    dump_list::Vector{String},
    dump_times::Vector{Float64},
    tA::Float64,
    tB::Float64,
    tf::Float64;
    t_obs_start::Union{Nothing,Float64} = nothing,
    t_obs_stop::Union{Nothing,Float64}  = nothing,
    output_dir::Union{Nothing,String}   = nothing,
    output_prefix::String               = "BriskImage",
    frame_index_mode::Symbol            = :time,
    pixel_stride::Int                   = 4,
    extrema_stride::Int                 = 1,
    n_window::Int                       = 3,
    anchor::Symbol                      = :mode,
    approach_bin::Bool                  = true,
    shadow_bin::Bool                    = true,
    band_ts::Union{Nothing,Vector{Vector{Float64}}} = nothing,
    ranges::Union{Nothing,Vector{NTuple{2,Float64}}} = nothing,
    step_radiating::Union{Nothing,Matrix{Vector{Bool}}} = nothing,
    step_bin::Union{Nothing,Matrix{Vector{Int8}}}       = nothing,
    tf_buffer::Union{Nothing,Float64}   = nothing,
    clamp_warn::Float64                 = 0.01,
    progress_every::Int                 = 512,
    verbose_windows::Bool               = false
)
    # ---- sanity -----------------------------------------------------------
    issorted(dump_times) || error("process_brisklight_images!: dump_times must be ascending.")
    length(dump_times) == length(dump_list) ||
        error("process_brisklight_images!: dump_times and dump_list length mismatch.")
    length(dump_times) >= n_window ||
        error("process_brisklight_images!: need at least n_window = $n_window dumps.")

    if hasproperty(model, :brisk_light)
        model.brisk_light ||
            @warn "model.brisk_light is false: time interpolation may be disabled for brisk light."
    else
        @warn "model has no `brisk_light` field: the local iharm.jl patch may be missing."
    end
    if hasproperty(model, :slow_light) && model.slow_light
        @warn "model.slow_light is true while rendering brisk light; flags are ambiguous."
    end

    dTs = diff(dump_times)
    dT  = length(dTs) > 0 ? dTs[1] : NaN
    if length(dTs) > 0 && !all(x -> isapprox(x, dT; rtol = 1e-6), dTs)
        @warn "dump_times are not uniformly spaced; W/dT diagnostics lose their meaning."
    end
    if n_window > 3
        @warn "n_window = $n_window requires the generalised set_tinterp_ns (bracket search over N snapshots). Without it, get_jk will only ever see the first three."
    end

    #### band statistics
    if band_ts === nothing || ranges === nothing
        band_ts, ranges = compute_band_modal_times!(
            midplane_crossings, all_geodesics, nsteps, pixels_x, pixels_y,
            params_brisklight, model, params_brisklight.p;
            pixel_stride = pixel_stride, extrema_stride = extrema_stride,
            approach_bin = approach_bin, shadow_bin = shadow_bin)
    else
        compute_band_modal_times!(params_brisklight, band_ts, ranges, params_brisklight.p)
    end

    # Per-step radiating flag and bin index: pure functions of geodesic geometry,
    # independent of t_obs and p, so they are computed once and reused for every
    # frame of this call and, if the caller passes them back in, for every p of
    # a sweep. This is what keeps the render loop free of `Iharm.bl_coord` calls;
    # see `precompute_step_info` for why that used to dominate the cost of a
    # sweep.
    if step_radiating === nothing || step_bin === nothing
        Rh_pc = 1.0 + sqrt(1.0 - model.a * model.a)
        step_radiating, step_bin = precompute_step_info(
            midplane_crossings, all_geodesics, nsteps, pixels_x, pixels_y,
            params_brisklight.n_bands, model, Rh_pc;
            approach_bin = approach_bin, shadow_bin = shadow_bin)
    end

    t_need_lo, t_need_hi = retained_support(params_brisklight)

    #### output
    outdir = output_dir === nothing ?
             joinpath("..", "..", "data", "Images", "BriskLight_p$(params_brisklight.p)") :
             output_dir
    mkpath(outdir)
    output_fmt   = joinpath(outdir, "$(output_prefix).%05d.txt")
    manifest_path = joinpath(outdir, "frames.csv")
    open(manifest_path, "w") do io
        println(io, "index,t_obs,file,clamped_fraction,n_loads")
    end

    # ---- schedule ---------------------------------------------------------
    buf     = tf_buffer === nothing ? model.rmax_geo : tf_buffer
    t_start = t_obs_start === nothing ? (tA - t_need_lo) : t_obs_start
    t_stop  = t_obs_stop  === nothing ? (min(tB, tf - buf) - t_need_hi) : t_obs_stop

    @info @sprintf("Brisk-light: p = %.3f  retained support = [%.3f, %.3f] M (width %.3f M, %.2f dT)",
                   params_brisklight.p, t_need_lo, t_need_hi,
                   t_need_hi - t_need_lo, (t_need_hi - t_need_lo) / dT)
    @info @sprintf("Brisk-light: observer grid t_obs = %.3f : %.3f : %.3f M",
                   t_start, params_brisklight.image_cadence, t_stop)

    if t_start > t_stop
        @warn "Brisk-light: empty observer window (t_start = $t_start > t_stop = $t_stop). Nothing to render."
        return (frames = 0, t_obs = Float64[], clamped = Float64[], loads = Int[],
                retained_support = (t_need_lo, t_need_hi),
                band_ts = band_ts, ranges = ranges,
                step_radiating = step_radiating, step_bin = step_bin)
    end

    ##### snapshot windows 
    # Probe-load one snapshot to fix the concrete element type, and keep it in
    # the cache so nothing is wasted.
    probe_idx  = first(find_dump_window(clamp(t_start + t_need_lo, dump_times[1], dump_times[end]),
                                        dump_times; n_window = n_window))
    probe_data = Iharm.load_data(dump_list[probe_idx], trat_large, model)
    D          = typeof(probe_data)
    dump_cache = Dict{Int,D}(probe_idx => probe_data)

    nb           = nbins(params_brisklight.n_bands)
    bin_windows  = [BandWindowState{D}() for _ in 1:nb]
    data_bins    = Vector{Vector{D}}(undef, nb)
    window_times = Vector{NTuple{2,Float64}}(undef, nb)
    for b in 1:nb
        data_bins[b]    = Vector{D}()
        window_times[b] = (Inf, -Inf)
    end

    Image      = zeros(Float64, pixels_x, pixels_y)
    bad_pixels = Threads.Atomic{Int}(0)
    prog_lock  = ReentrantLock()

    t_obs_log   = Float64[]
    clamp_log   = Float64[]
    load_log    = Int[]
    nimg        = 0
    last_img_target = t_start

    while last_img_target <= t_stop + 1e-9
        params_brisklight.t_obs = last_img_target
        n_loads = 0

        for b in 1:nb
            params_brisklight.band_valid[b] || continue
            t_anchor = band_anchor(params_brisklight, b; anchor = anchor) + last_img_target
            n_loads += update_band_window!(bin_windows[b], t_anchor, dump_times, dump_list,
                                           trat_large, model, dump_cache;
                                           n_window = n_window, verbose = verbose_windows)
            data_bins[b]    = bin_windows[b].snapshots
            window_times[b] = (bin_windows[b].t_window_start, bin_windows[b].t_window_end)
        end

        # Bins with too few samples have no statistics of their own; point them at
        # the first valid bin so that a stray pixel never meets an empty snapshot
        # vector. Their contribution is negligible by construction.
        fallback = findfirst(params_brisklight.band_valid)
        if fallback !== nothing
            for b in 1:nb
                params_brisklight.band_valid[b] && continue
                data_bins[b]    = data_bins[fallback]
                window_times[b] = window_times[fallback]
                params_brisklight.hdi_intervals[b] = params_brisklight.hdi_intervals[fallback]
            end
        end

        clamped, clamped_per_bin =
            availability_clamp_report(params_brisklight, band_ts, window_times, last_img_target)
        if isfinite(clamped) && clamped > clamp_warn
            @warn @sprintf("Brisk-light: frame t_obs = %.3f M has %.2f %% of steps hitting the availability clamp (frozen plasma). Per bin: %s",
                           last_img_target, 100 * clamped,
                           join([@sprintf("%s=%.1f%%", _bin_name(b, params_brisklight.n_bands),
                                          100 * clamped_per_bin[b])
                                 for b in 1:nb if isfinite(clamped_per_bin[b])], " "))
        end

        fill!(Image, 0.0)
        Threads.atomic_xchg!(bad_pixels, 0)

        p_bar = Progress(pixels_x * pixels_y;
                         desc = @sprintf("brisk p=%.2f t_obs=%.1f M ", params_brisklight.p, last_img_target),
                         showspeed = true, barlen = 30)
        done = Threads.Atomic{Int}(0)

        Threads.@threads :greedy for i in 1:pixels_x
            for j in 1:pixels_y
                integrate_brisklight_emission!(
                    all_geodesics[i, j], nsteps[i, j], Image, i, j,
                    freq, model.a, step_radiating[i, j], step_bin[i, j],
                    data_bins, params_brisklight.hdi_intervals, window_times,
                    last_img_target, model, bad_pixels)
            end
            d = Threads.atomic_add!(done, pixels_y) + pixels_y
            if d % progress_every < pixels_y
                lock(prog_lock) do
                    ProgressMeter.update!(p_bar, d)
                end
            end
        end
        finish!(p_bar)

        bad_pixels[] > 0 &&
            @warn "Brisk-light: $(bad_pixels[]) pixel(s) produced a non-finite intensity and were set to NaN."

        Image_out   = Image .* freq^3
        frame_index = frame_index_mode === :time ? round(Int, last_img_target) : nimg
        file_name   = Printf.format(Printf.Format(output_fmt), frame_index)
        writedlm(file_name, Image_out)

        open(manifest_path, "a") do io
            @printf(io, "%d,%.10f,%s,%.6f,%d\n",
                    frame_index, last_img_target, basename(file_name),
                    isfinite(clamped) ? clamped : NaN, n_loads)
        end

        push!(t_obs_log, last_img_target)
        push!(clamp_log, clamped)
        push!(load_log, n_loads)
        nimg += 1

        # Evict snapshots no longer referenced by any window.
        keep = Set{Int}()
        for b in 1:nb
            union!(keep, bin_windows[b].indices)
        end
        for k in collect(keys(dump_cache))
            k in keep || delete!(dump_cache, k)
        end

        last_img_target += params_brisklight.image_cadence
    end

    @info @sprintf("Brisk-light: %d frames, p = %.3f, mean clamped fraction = %.4f, total loads = %d, resident snapshots = %d",
                   nimg, params_brisklight.p,
                   isempty(clamp_log) ? NaN : mean(filter(isfinite, clamp_log)),
                   sum(load_log), length(dump_cache))
    @info "Brisk-light: manifest -> $manifest_path"

    return (frames = nimg, t_obs = t_obs_log, clamped = clamp_log, loads = load_log,
            retained_support = (t_need_lo, t_need_hi),
            band_ts = band_ts, ranges = ranges,
            step_radiating = step_radiating, step_bin = step_bin)
end

end # module Brisklight
