"""
Brisk-light rendering: slow-light geometry, reduced temporal fidelity.

Brisk light (Rojas-Paternina & Cardenas-Avendano, arXiv:2605.12659) evaluates
exactly the same geodesics, redshift factors and source positions as slow
light, and approximates only the *temporal* argument of the source. For each
lensing band `n` it builds the distribution of emission times over the screen,
as the paper does: each pixel's midplane-crossing time in that band (see
[`BandSample`](@ref)), takes the modal highest-density interval `T_{n,p}`
enclosing probability mass `p`, and clips the slow-light emission time into it
(Eq. V.2).

`p` is the one knob, and it controls the snapshot count *through the geometry*:
it sets the width of `T_{n,p}`, that width is fixed by the Kerr delay structure
and the extent of the emitting region, and the dump files band `n` needs are
then simply the ones bracketing the interval — about `ceil(width/dT) + 1` of
them. There is deliberately no "snapshots per band" parameter to set; see
[`band_dump_range`](@ref).

    p = 0    every step of band n is evaluated at the modal time t̄_n
    0 < p < 1  step times inside the interval are kept, the tails clip to its edges
    p = 1    the clip is the identity and what remains is slow light

That is the paper's clip, `interpolate_bands = false`. The default since Oct 21
is `interpolate_bands = true`: `p` still picks the snapshots each frame loads
(the band intervals above), but no step is clipped into its band; every step
keeps its true time, clamped to the loaded range, and reads the flow
interpolated between the loaded snapshots — so a step of band 1 can use band
0's snapshots and the time along a ray does not jump at the midplane crossing.
See [`build_timemap`](@ref).

# Synchronization

Band times are measured relative to band 0. A frame is labelled `t_frame`, the
absolute fluid time of its `n = 0` image — the time fast light would use — and
band `n` is evaluated around `t_frame + (t̄_n - t̄_0)`. So at `p = 0` band 0 sits
at exactly `t_frame`, and if `t_frame` is a dump time the interpolation weight
is 1 and the band uses that one dump, bitwise identically to fast light. Bands
`n > 0` generally land off the dump grid and interpolate between their two
nearest snapshots, which is both unavoidable and correct.

The consequence worth remembering when comparing runs: `t_frame` is the `n = 0`
clock, while slow light's target time is the *camera* clock. The two differ by
the constant `t̄_0`, so pair brisk frames with fast light by `t_frame` and with
slow light by `t_frame - t̄_0`. Both are recorded in the manifest.

# Usage

    series = DumpSeries(dump_paths)
    stats, cache = analyze_bands(midplane_crossings, all_geodesics, nsteps,
                                 nx, ny, n_bands, model, p)
    process_brisklight_images!(stats, cache, series, all_geodesics, nsteps, ...)
"""
module Brisklight

using Printf
using StaticArrays
using ProgressMeter
using ..Constants
using ..Coordinates
using ..GeoTypes
using ..Grid
using ..Radiation
using ..TimeInterp
using ..Utils
using ..Iharm
using ..Kharma
using ..Output
using ..Imaging
import ..Polarization

export BandStats, BandSample, StepCache, DumpSeries,
    band_walk, analyze_bands, band_stats, modal_hdi, weighted_quantile,
    retained_support, band_interval, band_dump_range, frame_dump_indices,
    build_timemap, slowlight_reference_timemap, clamped_fraction,
    integrate_brisk!, render_brisk_frame!, process_brisklight_images!,
    process_fastlight_images!,
    reader_for, dump_times_of

# ---------------------------------------------------------------------------
# Band convention
# ---------------------------------------------------------------------------
#
# `midplane_crossings[i,j]` is the per-pixel count of theta = pi/2 crossings
# returned by `Geodesics.get_pixel`: 0 for a captured ray, 1 for the direct
# image, 2 for the first indirect, and so on. It is offset by one from the
# lensing-band index n.
#
# Bands are assigned per *segment*, not per ray: walking a ray from its far end
# toward the camera, the segment bounded by crossings k+1 and k carries band k,
# so one ray feeds several bands and its time map changes as it traverses the
# plasma. The running counter is seeded at `midplane_crossings` and decremented
# at each crossing.
#
# The far-end tail, before the first crossing is reached, is capped at
# `midplane_crossings - 1` rather than being routed to a separate bin: that
# tail is the approach to the crossing that produces subimage
# `midplane_crossings - 1`, so it belongs to that band. Capping it is also what
# makes a direct ray (`midplane_crossings <= 1`) lie entirely in band 0, and
# hence reduce exactly to fast light at p = 0. Captured rays, which never
# cross, fall in band 0 for the same reason.
#
# Arrays are indexed `n + 1` for band n, n = 0 .. n_bands.

"""
    above_midplane(X)

Whether `X` lies above the equatorial plane.

Both the MKS and FMKS polar mappings send `X[3] = 1/2` to `theta = pi/2` and
are monotonic in `X[3]` there, so the sign of `X[3] - 1/2` is exactly the sign
of `theta - pi/2`. Testing `X[3]` directly therefore agrees with
`Coordinates.bl_coord` crossing for crossing while skipping its `exp` and `sin`
— which matters because this runs once per integration step per pixel.
"""
@inline above_midplane(X) = X[3] < 0.5

# ---------------------------------------------------------------------------
# Per-step cache
# ---------------------------------------------------------------------------

"""
    StepCache

Per-step band index and radiating flag for every pixel, in CSR layout.

Both quantities are pure functions of a geodesic's *spatial* coordinates, which
never change: only `X[1]` moves with the frame, and `p` changes the clip
interval rather than the band walk. So this is built once and reused for every
frame and every `p` of a sweep, which keeps the render loop down to two array
lookups per step.

The layout is flat on purpose. A `Matrix{Vector{Int8}}` is `nx*ny` separate
heap allocations that cannot be handed to a GPU; `band` and `radiating` here
are one allocation each and transfer as-is. Pixel `(i, j)`'s step `k` lives at
`offsets[pixel_index(c, i, j)] + k`.

# Fields
- `band::Vector{Int8}`: 0-based lensing-band index of the segment ending at
  `traj[k-1]`.
- `radiating::Vector{Bool}`: whether that segment lies inside the emitting shell.
- `offsets::Vector{Int32}`: CSR base offsets, length `nx*ny + 1`.
- `nx`, `ny`: image dimensions.
"""
struct StepCache
    band::Vector{Int8}
    radiating::Vector{Bool}
    offsets::Vector{Int32}
    nx::Int
    ny::Int
end

@inline pixel_index(c::StepCache, i::Integer, j::Integer) = (j - 1) * c.nx + i
@inline step_offset(c::StepCache, i::Integer, j::Integer) =
    @inbounds Int(c.offsets[pixel_index(c, i, j)])

"""
    BandSample{T}

The emission-time sample each band's statistics are built from, as defined in
arXiv:2605.12659 (Sec. III and V): for each pixel, the time at which its ray
crosses the equatorial plane inside the emitting region, one sample per pixel
per crossing, in the band of the crossing's order (the camera-nearest crossing
is n = 0), every sample counted once. For an equatorial disk that is exactly the
paper's per-band distribution of source times over the screen; for a volumetric
flow it is its geometric analogue, independent of any snapshot.

(Until Oct 24 the sample was every radiating integration step, weighted by its
emissivity in one reference snapshot. That distribution followed a single
snapshot's bright features, so its windows stalled at high inclination and
biased the flux; tested against this definition on one frame of the 4000 M MAD
runs, and dropped.)

# Fields
- `t[n+1]`, `w[n+1]`: the crossing times of band `n` and their weights (all 1;
  kept so the statistics stay generic in the weighting).
- `tmin[n+1]`, `tmax[n+1]`: support over *all* radiating steps of the band, not
  the crossings. The transfer loop reads a snapshot at every radiating step, so
  this, not the spread of crossing times, decides which dumps must be resident,
  and it is what `p = 1` opens the window to (slow light).
- `weighting`: `:pixels`.
"""
struct BandSample{T}
    t::Vector{Vector{T}}
    w::Vector{Vector{T}}
    tmin::Vector{T}
    tmax::Vector{T}
    weighting::Symbol
end

Base.length(s::BandSample) = length(s.t)

"""
    band_walk(midplane_crossings, all_geodesics, nsteps, nx, ny, n_bands, model, Rh;
              sample_stride = 1, grouping = :nearest_crossing)

Walk every geodesic once and return `(cache, sample::BandSample)`.

One pass does all three jobs: the per-step band and radiating flags of the
render cache, the crossing-time sample the KDE is built from (see
[`BandSample`](@ref)), and the support of the radiating steps that decides which
dumps a frame needs. Walking once matters because the non-radiating outbound leg
(`r: rmax_geo .. ro`) is the bulk of most rays' steps.

Crossings are counted on every step, inside or outside the emitting shell, so
the crossing order tracks `Geodesics.trace_geodesic`; only crossings inside the
emitting region (`Radiation.radiating_region` at the step's end) enter the
sample, as the paper's finite emitting disk. A crossing's time is the midpoint
of the step it falls in.

# Keywords
- `sample_stride`: screen subsampling for the crossing sample (every pixel by
  default; one sample per crossing is cheap). `tmin`/`tmax` are collected at
  full resolution regardless.
- `grouping`: which band each *step* is rendered in (it only matters with the
  per-band clip, `interpolate_bands = false`). `:nearest_crossing` (default)
  gives a step the order of the crossing nearest to it along the ray, the same
  order the sample uses; `:segment` gives the segment between crossings k and
  k+1 band k+1, as described under "Band convention".
"""
function band_walk(midplane_crossings::AbstractMatrix{Int},
                   all_geodesics,
                   nsteps::AbstractMatrix{Int},
                   nx::Int, ny::Int,
                   n_bands::Int,
                   model,
                   Rh;
                   sample_stride::Int = 1,
                   grouping::Symbol = :nearest_crossing)

    T = Float64
    grouping in (:segment, :nearest_crossing) ||
        error("band_walk: grouping must be :segment or :nearest_crossing, got :$grouping")
    nearest = grouping === :nearest_crossing
    npix = nx * ny
    nb = n_bands + 1

    # CSR offsets first, so every thread knows where to write.
    offsets = Vector{Int32}(undef, npix + 1)
    offsets[1] = 0
    total = 0
    for j in 1:ny, i in 1:nx
        total += max(nsteps[i, j], 1)
        offsets[(j - 1) * nx + i + 1] = Int32(total)
    end

    band = Vector{Int8}(undef, total)
    radiating = Vector{Bool}(undef, total)

    # Per-row accumulators, merged after the loop: no locks in the walk, and no
    # reliance on `threadid()`, which under `:greedy` scheduling can exceed
    # `nthreads()` (the interactive pool has its own ids) and is not a stable
    # identity for a task anyway. The row index is unique per iteration, so it is
    # the safe thing to key on.
    tsamp = [[T[] for _ in 1:nb] for _ in 1:nx]
    tlo = [fill(T(Inf), nb) for _ in 1:nx]
    thi = [fill(T(-Inf), nb) for _ in 1:nx]

    Threads.@threads :greedy for i in 1:nx
        my_samp = tsamp[i]
        my_lo = tlo[i]
        my_hi = thi[i]
        sample_col = (i - 1) % sample_stride == 0
        scross = T[]   # arc length of each midplane crossing of the current ray, far -> near

        for j in 1:ny
            n = nsteps[i, j]
            base = Int(offsets[(j - 1) * nx + i])

            if n < 2
                @inbounds band[base + 1] = Int8(0)
                @inbounds radiating[base + 1] = false
                continue
            end

            traj = all_geodesics[i, j]
            seed = midplane_crossings[i, j]
            # Cap for the far-end tail: the band its first crossing produces.
            bmax = min(max(seed - 1, 0), n_bands)
            current = seed
            above_prev = above_midplane(traj[n].X)
            take_sample = sample_col && (j - 1) % sample_stride == 0

            # First pass: where along the ray (arc length from the far end) each midplane crossing lies, and how
            # many there are, so each crossing knows its order counted from the camera (the camera-nearest is 0).
            empty!(scross)
            let ab = above_prev, sarc = zero(T)
                for nstep in n:-1:2
                    @inbounds stp = traj[nstep - 1]
                    a2 = above_midplane(stp.X)
                    if a2 != ab
                        push!(scross, sarc + T(0.5) * T(stp.dl))
                        ab = a2
                    end
                    sarc += T(stp.dl)
                end
            end
            ncr = length(scross)
            mcr = 1; sarc = zero(T); kcr = 0

            for nstep in n:-1:2
                @inbounds Xf = traj[nstep - 1].X
                rad = Radiation.radiating_region(Xf, model, Rh)

                above_curr = above_midplane(Xf)
                if above_curr != above_prev
                    current = max(current - 1, 0)
                    above_prev = above_curr
                    kcr += 1
                    # the sample: this crossing's time, in the band of its order, if it lies in the emitting region
                    if take_sample && rad
                        @inbounds push!(my_samp[min(ncr - kcr, n_bands) + 1], T(0.5) * (traj[nstep].X[1] + Xf[1]))
                    end
                end

                # :nearest_crossing (Oct 21 audit): a step belongs to the lensing order of the midplane crossing
                # nearest to it in arc length along the ray, the boundaries halfway between neighbouring crossings, so
                # both sides of an emitting layer get the same band. :segment gives the far side of each crossing the
                # next band, splitting the layer between two epochs.
                b = if nearest
                    @inbounds sarc += T(traj[nstep - 1].dl)   # arc length at Xf, where the step is evaluated
                    if ncr == 0
                        0
                    else
                        @inbounds while mcr < ncr && abs(scross[mcr + 1] - sarc) <= abs(scross[mcr] - sarc)
                            mcr += 1
                        end
                        min(ncr - mcr, n_bands)
                    end
                else
                    min(current, bmax)
                end
                @inbounds band[base + nstep] = Int8(b)
                @inbounds radiating[base + nstep] = rad

                # The support: the times of the band's radiating steps, which are the steps that read a snapshot.
                if rad
                    t = Xf[1]
                    bi = b + 1
                    @inbounds t < my_lo[bi] && (my_lo[bi] = t)
                    @inbounds t > my_hi[bi] && (my_hi[bi] = t)
                end
            end
            # Step 1 is the camera endpoint; it seeds nothing but must be defined.
            @inbounds band[base + 1] = band[base + 2]
            @inbounds radiating[base + 1] = false
        end
    end

    st = [T[] for _ in 1:nb]
    tmin = fill(T(Inf), nb)
    tmax = fill(T(-Inf), nb)
    for b in 1:nb
        sizehint!(st[b], sum(length(tsamp[i][b]) for i in 1:nx))
        for i in 1:nx
            append!(st[b], tsamp[i][b])
            tmin[b] = min(tmin[b], tlo[i][b])
            tmax[b] = max(tmax[b], thi[i][b])
        end
    end

    cache = StepCache(band, radiating, offsets, nx, ny)
    return cache, BandSample{T}(st, [ones(T, length(x)) for x in st], tmin, tmax, :pixels)
end

# ---------------------------------------------------------------------------
# Modal highest-density interval
# ---------------------------------------------------------------------------

"""
    quantile_sorted(xs, q)

Linearly interpolated quantile of an already-sorted vector. Written out rather
than pulled from a statistics package so this module adds no dependency to
Jipole and stays generic in the element type.
"""
function quantile_sorted(xs::AbstractVector{T}, q::Real) where {T}
    n = length(xs)
    n == 0 && return T(NaN)
    n == 1 && return xs[1]
    h = (n - 1) * clamp(float(q), 0.0, 1.0) + 1
    lo = floor(Int, h)
    hi = min(lo + 1, n)
    frac = h - lo
    return xs[lo] + T(frac) * (xs[hi] - xs[lo])
end

"""
    weighted_quantile(xs_sorted, ws, q)

Inverse empirical-CDF quantile of a weighted sample sorted by time. Returning
an observed time preserves CDF jumps: interpolating across a dominant atom and
then filtering by time can discard that atom, even at the 99.75% quantile.
Used to trim the KDE's domain on the weighted distribution rather than the raw
one, so a long tail of near-zero-emissivity steps cannot stretch the grid.
"""
function weighted_quantile(xs::AbstractVector{T}, ws::AbstractVector{T}, q::Real) where {T}
    n = length(xs)
    n == 0 && return T(NaN)
    n == 1 && return xs[1]
    total = sum(ws)
    total > 0 || return xs[1]
    target = T(clamp(float(q), 0.0, 1.0)) * total
    acc = zero(T)
    @inbounds for i in 1:n
        acc += ws[i]
        acc >= target && return xs[i]
    end
    return xs[n]
end

"""
    silverman_bandwidth(xs, ws = nothing)

Silverman's rule-of-thumb kernel width, using the smaller of the standard
deviation and the interquartile range so a heavy tail does not oversmooth the
mode.
"""
function silverman_bandwidth(xs::AbstractVector{T}, ws::Union{Nothing,AbstractVector{T}} = nothing) where {T}
    n = length(xs)
    n < 2 && return one(T)
    if ws === nothing
        m = sum(xs) / n
        s = sqrt(max(sum(x -> (x - m)^2, xs) / (n - 1), zero(T)))
        neff = T(n)
    else
        # KDE weights have arbitrary physical units and can be extremely small (invariant emissivity * affine
        # length). Normalize their scale before taking moments: an absolute eps floor on sum(w^2) changes the
        # effective sample size under a harmless change of units (it collapsed to its floor of 2 for the j*dl
        # weights, inflating the bandwidth several-fold; Oct 21 audit).
        wscale = maximum(ws)
        wscale > zero(T) || return one(T)
        vw = ws ./ wscale
        sw = sum(vw)
        m = sum(xs[i] * vw[i] for i in eachindex(xs)) / sw
        s = sqrt(max(sum(vw[i] * (xs[i] - m)^2 for i in eachindex(xs)) / sw, zero(T)))
        # Kish effective sample size, so a few dominant weights do not pretend to the resolution of the full
        # sample. One scaled weight equals one, so the denominator is positive without an epsilon.
        neff = sw * sw / sum(w -> w * w, vw)
    end
    # The same probability measure as the KDE: counting near-zero-weight steps in the IQR would let the
    # integrator's step placement set the bandwidth.
    ord = sortperm(xs)
    sx = xs[ord]
    iqr = if ws === nothing
        quantile_sorted(sx, 0.75) - quantile_sorted(sx, 0.25)
    else
        sws = ws[ord]
        weighted_quantile(sx, sws, 0.75) - weighted_quantile(sx, sws, 0.25)
    end
    a = iqr > 0 ? min(s, iqr / T(1.349)) : s
    a <= 0 && return one(T)
    return T(0.9) * a * max(neff, T(2))^T(-0.2)
end

"""
    modal_hdi(ts, p; gridsize, trim, bandwidth)

Modal highest-density interval of `ts` at probability mass `p`.

Returns `(mode, lo, hi, mass, bandwidth)`. `p = 0` returns the mode as a
degenerate interval; `p = 1` returns `(-Inf, Inf)`, bypassing the clip entirely
rather than returning the trimmed quantiles — clipping the tails at `p = 1`
would stop brisk light converging to slow light, which is the whole point of
that limit.

The density is a *binned* Gaussian KDE: the samples are histogrammed onto the
grid and convolved with a kernel truncated at four bandwidths. Direct
evaluation would be `O(gridsize x length(ts))`, and `ts` here holds one entry
per integration step per sampled pixel — millions. Binning makes it
`O(length(ts) + gridsize x kernel_width)` with no practical loss, and keeps the
module free of an FFT dependency.

The interval is the connected component of `{density >= threshold}` containing
the global mode, with `threshold` bisected until the component encloses mass
`p`. Anchoring on the mode rather than on a central interval matters because
delay distributions are strongly asymmetric.

`interval = :shortest` (Oct 28) keeps the KDE mode but takes instead the shortest
interval of the trimmed sample that contains the mode and weight `p`. For a
single-peaked distribution that is the same interval; where the density dips and
rises again (the direct image at 80 degrees: a peak, then a flat shelf of outer
pixels taller than the dip), the threshold's component jumps from ~48% to ~97%
of the pixels at once, and no threshold encloses 60%, while the shortest
interval grows continuously with `p`.
"""
function modal_hdi(ts::AbstractVector{T}, p::Real;
                   weights::Union{Nothing,AbstractVector{T}} = nothing,
                   gridsize::Int = 2048,
                   trim::Tuple{Float64,Float64} = (0.0025, 0.9975),
                   bandwidth::Union{Nothing,T} = nothing,
                   interval::Symbol = :hdi,
                   anchor::Union{Nothing,Real} = nothing) where {T<:AbstractFloat}

    pp = T(p)
    (zero(T) <= pp <= one(T)) || error("modal_hdi: p must be in [0, 1], got $p")
    interval in (:hdi, :shortest) || error("modal_hdi: interval must be :hdi or :shortest, got :$interval")
    # anchor (Oct 31): the time the interval is built around instead of the KDE mode, e.g. fast light's emission-
    # weighted modal time for band 0, so brisk and fast light share a clock. A density threshold's component is
    # defined by the mode, so only the shortest interval (or p = 0, the anchor itself) can take one.
    (anchor === nothing || interval === :shortest || pp == zero(T)) ||
        error("modal_hdi: an anchor needs interval = :shortest (or p = 0)")

    # Keep times and weights together while filtering, then sort by time so the
    # quantiles below are meaningful.
    keep = weights === nothing ?
           [i for i in eachindex(ts) if isfinite(ts[i])] :
           [i for i in eachindex(ts) if isfinite(ts[i]) && isfinite(weights[i]) && weights[i] > 0]
    length(keep) >= 2 || error("modal_hdi: need at least 2 usable samples, got $(length(keep))")
    ord = sortperm(view(ts, keep))
    idx = keep[ord]
    xs = T[ts[i] for i in idx]
    ws = weights === nothing ? nothing : T[weights[i] for i in idx]

    # Trim on the *weighted* distribution, so a long tail of near-zero-weight
    # steps cannot set the KDE's domain.
    lo_t, hi_t = if ws === nothing
        quantile_sorted(xs, trim[1]), quantile_sorted(xs, trim[2])
    else
        weighted_quantile(xs, ws, trim[1]), weighted_quantile(xs, ws, trim[2])
    end
    if !(lo_t < hi_t)
        lo_t, hi_t = xs[1], xs[end]
    end
    lo_t < hi_t || error("modal_hdi: degenerate sample (all values equal $(xs[1]))")

    # The paper's KDE is fitted to the trimmed finite sample, so the bandwidth is fitted to that same sample
    # rather than to the discarded tails.
    trim_idx = findall(x -> lo_t <= x <= hi_t, xs)
    length(trim_idx) >= 2 || error("modal_hdi: too few samples after trimming")
    xs = xs[trim_idx]
    ws = ws === nothing ? nothing : ws[trim_idx]
    h = bandwidth === nothing ? silverman_bandwidth(xs, ws) : bandwidth
    (isfinite(h) && h > zero(T)) || error("modal_hdi: bandwidth must be finite and positive")

    # Pad the domain so the kernel does not pile mass up against the edges.
    pad = 3 * h
    gmin = lo_t - pad
    gmax = hi_t + pad
    ng = max(gridsize, 16)
    dx = (gmax - gmin) / (ng - 1)

    counts = zeros(T, ng)
    wtot = zero(T)
    nin = 0
    for (k, x) in pairs(xs)
        (lo_t <= x <= hi_t) || continue
        bin = round(Int, (x - gmin) / dx) + 1
        (1 <= bin <= ng) || continue
        wk = ws === nothing ? one(T) : ws[k]
        @inbounds counts[bin] += wk
        wtot += wk
        nin += 1
    end
    (nin >= 2 && wtot > 0) || error("modal_hdi: too little weight after trimming ($nin samples, total weight $wtot)")

    half = max(1, ceil(Int, 4 * h / dx))
    kern = Vector{T}(undef, 2 * half + 1)
    inv2h2 = one(T) / (2 * h * h)
    ksum = zero(T)
    for k in -half:half
        v = exp(-(k * dx)^2 * inv2h2)
        kern[k + half + 1] = v
        ksum += v
    end
    kern ./= ksum

    density = zeros(T, ng)
    for i in 1:ng
        @inbounds ci = counts[i]
        ci == 0 && continue
        for k in max(1, i - half):min(ng, i + half)
            @inbounds density[k] += ci * kern[k - i + half + 1]
        end
    end
    density ./= (wtot * dx)

    mode_idx = argmax(density)
    t_mode = anchor === nothing ? gmin + (mode_idx - 1) * dx : T(anchor)

    pp == zero(T) && return (mode = t_mode, lo = t_mode, hi = t_mode,
                             mass = zero(T), bandwidth = h)
    pp == one(T) && return (mode = t_mode, lo = T(-Inf), hi = T(Inf),
                            mass = one(T), bandwidth = h)

    if interval === :shortest
        # Two pointers over the sorted trimmed sample: for each left end at or before the mode, the first right end at
        # or after it whose cumulative weight reaches p; that right end only moves forward as the left end does.
        n = length(xs)
        cw = ws === nothing ? collect(T, 1:n) : cumsum(ws)
        target = pp * cw[end]
        jm = clamp(searchsortedfirst(xs, t_mode), 1, n)
        imax = max(searchsortedlast(xs, t_mode), 1)
        best_lo, best_hi, best_w = xs[1], xs[n], cw[n]
        j = jm
        for i in 1:imax
            j = max(j, i, jm)
            below = i > 1 ? cw[i - 1] : zero(T)
            while j < n && cw[j] - below < target
                j += 1
            end
            cw[j] - below < target && break
            if xs[j] - xs[i] < best_hi - best_lo
                best_lo, best_hi, best_w = xs[i], xs[j], cw[j] - below
            end
        end
        return (mode = t_mode, lo = best_lo, hi = best_hi, mass = best_w / cw[end], bandwidth = h)
    end

    cmass = cumsum(density) .* dx
    @inline component_mass(l, r) = cmass[r] - (l > 1 ? cmass[l - 1] : zero(T))

    function modal_component(thresh)
        l = mode_idx
        r = mode_idx
        @inbounds while l > 1 && density[l - 1] >= thresh
            l -= 1
        end
        @inbounds while r < ng && density[r + 1] >= thresh
            r += 1
        end
        return l, r
    end

    low = zero(T)
    high = density[mode_idx]
    for _ in 1:60
        mid = (low + high) / 2
        l, r = modal_component(mid)
        if component_mass(l, r) >= pp
            low = mid
        else
            high = mid
        end
    end
    l, r = modal_component(low)

    return (mode = t_mode,
            lo = gmin + (l - 1) * dx,
            hi = gmin + (r - 1) * dx,
            mass = component_mass(l, r),
            bandwidth = h)
end

# ---------------------------------------------------------------------------
# Band statistics
# ---------------------------------------------------------------------------

"""
    BandStats{T}

Per-band emission-time statistics at a given `p`, all in *geometric* time
(negative, with the camera at 0).

# Fields
- `n_bands`: highest band index tracked.
- `modal[n+1]`: `t̄_n`, the band's modal emission time.
- `lo[n+1]`, `hi[n+1]`: the retained interval `T_{n,p}`.
- `tmin[n+1]`, `tmax[n+1]`: the full sampled support, independent of `p`.
- `valid[n+1]`: false for bands with too few samples to have statistics; they
  inherit band 0's interval and contribute negligibly by construction.
- `nsamples[n+1]`, `bandwidth[n+1]`: diagnostics.
- `p`: the retained probability mass these statistics were built at.
- `t_ref`: `t̄_0`, the synchronization reference. Every band's time is measured
  relative to this, which is what pins `n = 0` to the fast-light snapshot.
- `support`: the union of every band's sampled support. Band intervals are
  intersected with this rather than with each band's own support, which is what
  makes `p = 1` collapse to a single band-independent interval -- see
  [`band_interval`](@ref).
"""
struct BandStats{T}
    n_bands::Int
    modal::Vector{T}
    lo::Vector{T}
    hi::Vector{T}
    tmin::Vector{T}
    tmax::Vector{T}
    valid::Vector{Bool}
    nsamples::Vector{Int}
    bandwidth::Vector{T}
    p::T
    t_ref::T
    support::NTuple{2,T}
end

nbands(s::BandStats) = s.n_bands
nbins(s::BandStats) = s.n_bands + 1

# A band can be absent from the full-resolution emitting geometry even when n_bands requested it. Distinguish this
# from an active but unresolved KDE (Oct 21 audit): an absent band needs no snapshots at all.
@inline has_band_support(s::BandStats, b::Int) =
    isfinite(s.tmin[b]) && isfinite(s.tmax[b]) && s.tmin[b] <= s.tmax[b]

"""
    band_stats(sample::BandSample, p; min_samples = 32, verbose = true, kw...)

Build a [`BandStats`](@ref) from a sample collected by [`band_walk`](@ref).

Separated from the walk so a `p` sweep pays for the geodesics once: pass the
same `sample` back in for every `p`.

`t_ref0` (optional): band 0's anchor and so the synchronization reference `t_ref`,
in place of its modal time; e.g. fast light's emission-weighted modal time, so that
brisk frame `k` renders the direct image at fast light's time. Band 0's window is
then built around it (`interval = :shortest`); bands `n >= 1` keep their own modes.
"""
function band_stats(sample::BandSample{T}, p::Real;
                    min_samples::Int = 32,
                    verbose::Bool = true,
                    t_ref0::Union{Nothing,Real} = nothing,
                    kw...) where {T<:AbstractFloat}

    nb = length(sample)
    n_bands = nb - 1
    tmin = sample.tmin
    tmax = sample.tmax
    modal = fill(T(NaN), nb)
    lo = fill(T(NaN), nb)
    hi = fill(T(NaN), nb)
    valid = fill(false, nb)
    nsamp = zeros(Int, nb)
    bw = fill(T(NaN), nb)

    for b in 1:nb
        ts = sample.t[b]
        wsb = sample.w[b]
        nsamp[b] = length(ts)
        if length(ts) >= min_samples && maximum(ts) > minimum(ts)
            # t_ref0 (Oct 31): band 0's anchor, the synchronization reference, given instead of its modal time
            r = modal_hdi(ts, p; weights = wsb, anchor = b == 1 ? t_ref0 : nothing, kw...)
            modal[b] = r.mode
            lo[b] = r.lo
            hi[b] = r.hi
            bw[b] = r.bandwidth
            valid[b] = true
        elseif !isempty(ts)
            @warn """
                Brisk light: band n=$(b-1) has only $(length(ts)) weighted samples (need                 $min_samples), so it has no emission-time statistics and will be rendered                 *unclipped*, i.e. as slow light. That is the accurate choice, but it costs                 the band the full snapshot support, so one unresolved band can wipe out the                 I/O saving for the whole frame. Either lower n_bands to $(b-2), raise the                 image resolution, or lower sample_stride."""
        end
    end

    valid[1] || error("""
        Brisk light: band n=0 has no statistics ($(nsamp[1]) samples), so there is no
        synchronization reference. Lower `sample_stride`, raise the resolution, or check
        that the camera actually sees the emitting region.""")

    t_ref = modal[1]
    sup_lo = minimum(x for x in tmin if isfinite(x))
    sup_hi = maximum(x for x in tmax if isfinite(x))
    stats = BandStats{T}(n_bands, modal, lo, hi, copy(tmin), copy(tmax),
                         valid, nsamp, bw, T(p), t_ref, (T(sup_lo), T(sup_hi)))

    if verbose
        @info @sprintf("Brisk light: p = %.3f, %s sample, %s interval, synchronization reference t̄_0 = %.4f M%s",
                       p, sample.weighting, get(kw, :interval, :hdi), t_ref,
                       t_ref0 === nothing ? "" : " (given, not band 0's modal time)")
        nbad = count(!, valid)
        nbad > 0 && @info @sprintf("Brisk light: %d of %d band(s) unresolved and rendered unclipped; snapshots per image will not fall with p until they are resolved.",
                                   nbad, nb)
        for b in 1:nb
            valid[b] || continue
            w = hi[b] - lo[b]
            @info @sprintf("  n=%d: t̄ = %10.4f M  (offset %+9.4f)  T_p = [%.4f, %.4f]  width %s  support [%.4f, %.4f]  (%d samples)",
                           b - 1, modal[b], modal[b] - t_ref, lo[b], hi[b],
                           isfinite(w) ? @sprintf("%.4f M", w) : "inf",
                           tmin[b], tmax[b], nsamp[b])
        end
    end
    return stats
end

"""
    analyze_bands(midplane_crossings, all_geodesics, nsteps, nx, ny, n_bands,
                  model, p; kw...)

Convenience wrapper: walk the geodesics and build the statistics in one call.
Returns `(stats, cache, sample)`; keep `cache` and `sample` to sweep `p`
without re-walking the geodesics — call [`band_stats`](@ref) again on the same
`sample` at the new `p`.
"""
function analyze_bands(midplane_crossings, all_geodesics, nsteps,
                       nx::Int, ny::Int, n_bands::Int, model, p::Real;
                       sample_stride::Int = 1, min_samples::Int = 32, verbose::Bool = true,
                       grouping::Symbol = :nearest_crossing, kw...)
    Rh = 1 + sqrt(1.0 - model.a * model.a)
    cache, sample = band_walk(midplane_crossings, all_geodesics, nsteps,
                              nx, ny, n_bands, model, Rh;
                              sample_stride = sample_stride, grouping = grouping)
    stats = band_stats(sample, p; min_samples = min_samples, verbose = verbose, kw...)
    return stats, cache, sample
end

"""
    band_interval(stats, b) -> (lo, hi)

Band `b`'s retained interval in geometric time.

The interval is the band's HDI intersected with `stats.support`, the union of
every band's sampled support. The intersection stops the KDE's padded domain
from demanding dumps no ray needs; using the *union* rather than band `b`'s own
support is what preserves the `p = 1` limit, and both details matter:

- At `p = 1` the HDI is `(-Inf, Inf)`, so every band's interval becomes the same
  `stats.support`. The map loses all band dependence and is exactly slow light's
  rigid shift over that window — see [`slowlight_reference_timemap`](@ref).
  Intersecting with each band's own support instead would leave a residual
  band-dependent clip, and `p = 1` would never quite reach slow light.
- A band with too few samples to have statistics is left unclipped for the same
  reason. That renders it as slow light, which is the accurate choice; giving it
  another band's interval would clip its steps to times they never occupy, and
  since such a band is the faint high-`n` one, the error would land exactly
  where the band structure is supposed to live.
"""
@inline function band_interval(s::BandStats{T}, b::Int) where {T}
    sup_lo, sup_hi = s.support
    s.valid[b] || return (sup_lo, sup_hi)
    lo = max(s.lo[b], sup_lo)
    hi = min(s.hi[b], sup_hi)
    return hi >= lo ? (lo, hi) : (s.modal[b], s.modal[b])
end

"""
    retained_support(stats) -> (t_lo, t_hi)

Earliest and latest geometric source time any frame needs at this `p`, reduced
over bands with radiating support, including unresolved active bands. When all
active bands are resolved, `p = 0` collapses onto the modal times; `p = 1` is
the full sampled support. This, not the modal times, is what bounds the
observer-time schedule and how many dumps must be resident — and it shrinks
with `p`, so a low-`p` run can cover a longer observer window from the same
dumps than slow light can.
"""
function retained_support(s::BandStats{T}) where {T}
    t_lo = T(Inf)
    t_hi = T(-Inf)
    for b in 1:nbins(s)
        has_band_support(s, b) || continue
        lo, hi = band_interval(s, b)
        isfinite(lo) && (t_lo = min(t_lo, lo))
        isfinite(hi) && (t_hi = max(t_hi, hi))
    end
    (isfinite(t_lo) && isfinite(t_hi)) || error("retained_support: no radiating band")
    return t_lo, t_hi
end


"""
    band_dump_range(lo_abs, hi_abs, dump_times) -> UnitRange{Int}

The dumps needed to interpolate over the absolute time interval
`[lo_abs, hi_abs]`: those bracketing it.

This is where `p` turns into I/O, and nothing else decides it:

| interval                      | dumps                        |
|:------------------------------|:-----------------------------|
| a point, on a dump time       | 1 — no interpolation at all  |
| a point, between dumps        | 2 — the two nearest          |
| width `w`                     | about `ceil(w/dT) + 1`       |

An interval running past either end of the series clamps to the terminal dump,
which freezes the plasma there rather than extrapolating it; the fraction of
steps that happens to is reported by [`clamped_fraction`](@ref).
"""
function band_dump_range(lo_abs::Real, hi_abs::Real, dump_times::AbstractVector)
    n = length(dump_times)
    iA = clamp(searchsortedlast(dump_times, lo_abs), 1, n)
    iB = clamp(searchsortedfirst(dump_times, hi_abs), 1, n)
    iB < iA && (iB = iA)
    return iA:iB
end

"""
    frame_dump_indices(stats, t_frame, dump_times) -> (indices, per_band)

Dump indices a frame at `t_frame` needs, as the sorted union over bands, plus
the per-band ranges for diagnostics.

Band `n`'s interval is shifted by `t_frame - t̄_0`, which is the
synchronization: band 0 lands on `t_frame`, every other band keeps its physical
offset from band 0.
"""
function frame_dump_indices(s::BandStats{T}, t_frame::Real,
                            dump_times::AbstractVector) where {T}
    shift = T(t_frame) - s.t_ref
    per_band = Vector{UnitRange{Int}}(undef, nbins(s))
    idx = Int[]
    for b in 1:nbins(s)
        if !has_band_support(s, b)
            per_band[b] = 1:0
            continue
        end
        lo, hi = band_interval(s, b)
        r = band_dump_range(lo + shift, hi + shift, dump_times)
        per_band[b] = r
        for k in r
            k in idx || push!(idx, k)
        end
    end
    sort!(idx)
    return idx, per_band
end

"""
    build_timemap(stats, t_frame, tA, tB; interpolate_bands = true) -> TimeMap

The frame's time map: the synchronization reference `t̄_0`, the availability
clamp `[tA, tB]` set by the first and last snapshot actually loaded, and either
no band clip (the default) or the per-band clip bounds in geometric time.

`interpolate_bands = true` (the default since Oct 21): the bands only decide
which snapshots a frame loads (`frame_dump_indices`); every step keeps its true
time, clamped to `[tA, tB]`, and reads the flow interpolated between the loaded
snapshots that bracket it. Inside a band's interval that is slow light; in a
gap between two bands' intervals it is the linear blend of the two nearest
loaded snapshots. Along a ray the time is then continuous, where the clip makes
it jump at every midplane crossing — the band boundary, which in a volumetric
flow runs through the brightest emission (the equatorial disk of
arXiv:2605.12659 emits only at the crossing, so there the jump does no harm).
Once the band intervals overlap (p = 0.6 in the 1000 M MAD production, but not
p = 0.3) the loaded snapshots are contiguous and every step inside `[tA, tB]`
is exactly slow light, at no extra I/O.

`interpolate_bands = false` is the paper's prescription (Eq. V.2): each step is
clipped into its own band's retained interval `T_{n,p}`. `p = 0` then
reproduces fast light on direct-image pixels bitwise, which `interpolate_bands`
does not.
"""
function build_timemap(s::BandStats{T}, t_frame::Real, tA::Real, tB::Real;
                       interpolate_bands::Bool = true) where {T}
    if interpolate_bands
        R = promote_type(T, typeof(float(t_frame)), typeof(float(tA)), typeof(float(tB)))
        nb = nbins(s)
        return TimeMap{nb,R}(R(t_frame), R(s.t_ref), SVector{nb,R}(fill(R(-Inf), nb)),
                             SVector{nb,R}(fill(R(Inf), nb)), R(tA), R(tB))
    end
    # Promote rather than taking the element type from the statistics: the frame
    # time is a legitimate differentiation variable (fitting a light curve's
    # phase, say), and the statistics are always Float64 because they come from a
    # KDE over sampled times. Pinning the map to the statistics' type would make
    # a Dual frame time a MethodError.
    R = promote_type(T, typeof(float(t_frame)), typeof(float(tA)), typeof(float(tB)))
    nb = nbins(s)
    los = Vector{R}(undef, nb)
    his = Vector{R}(undef, nb)
    for b in 1:nb
        lo, hi = band_interval(s, b)
        los[b] = lo
        his[b] = hi
    end
    return TimeMap{nb,R}(R(t_frame), R(s.t_ref),
                         SVector{nb,R}(los), SVector{nb,R}(his),
                         R(tA), R(tB))
end

"""
    slowlight_reference_timemap(stats, t_frame, tA, tB) -> TimeMap

The slow-light map on brisk light's clock: one band, no clipping beyond
`stats.support`, and the same `t_frame - t̄_0` shift the brisk map applies.

This is the reference a `p = 1` run must reproduce *bitwise*. Rendering it
through [`render_brisk_frame!`](@ref) with the same geodesics, the same band
cache and the same [`TimeInterp.SnapshotSet`](@ref) as the brisk run leaves the
time map as the only difference, which is what makes the `p -> 1` limit testable
as an identity rather than as a tolerance.

Note the clock: `t_frame` is brisk light's `n = 0` time, while upstream
`Slowlight` schedules on the camera time. The two differ by `t̄_0`, so a frame
here corresponds to slow light's target time `t_frame - stats.t_ref`.
"""
function slowlight_reference_timemap(s::BandStats{T}, t_frame::Real,
                                     tA::Real, tB::Real) where {T}
    R = promote_type(T, typeof(float(t_frame)), typeof(float(tA)), typeof(float(tB)))
    sup_lo, sup_hi = s.support
    return TimeMap{1,R}(R(t_frame), R(s.t_ref),
                        SVector{1,R}(sup_lo), SVector{1,R}(sup_hi), R(tA), R(tB))
end

"""
    clamped_fraction(stats, sample, t_frame, tA, tB) -> (total, per_band)

Fraction of sampled steps whose clipped time falls outside `[tA, tB]` and is
therefore evaluated against a frozen plasma.

This is the correctness gate on a brisk run, not a performance number: a run
whose clamped fraction is not negligible is not testing brisk light, and must
not be compared against slow light.

It is computed with the band clip whatever `interpolate_bands` is, so it asks
whether the dump series covers every band's interval. Under
`interpolate_bands`, steps whose true time lies outside the loaded range are
held at its edge by design (that replaces the clip), and are not counted here.
"""
function clamped_fraction(s::BandStats{T}, sample::BandSample{T},
                          t_frame::Real, tA::Real, tB::Real) where {T}
    nb = nbins(s)
    per_band = fill(T(NaN), nb)
    ntot = 0
    nclamp = 0
    shift = T(t_frame) - s.t_ref
    for b in 1:nb
        has_band_support(s, b) || continue
        ts = sample.t[b]
        isempty(ts) && continue
        lo, hi = band_interval(s, b)
        c = 0
        for t in ts
            ta = clamp(t + shift, lo + shift, hi + shift)
            (ta < tA || ta > tB) && (c += 1)
        end
        per_band[b] = c / length(ts)
        ntot += length(ts)
        nclamp += c
    end
    return (ntot == 0 ? T(NaN) : T(nclamp / ntot)), per_band
end


"""
    reader_for(path)

The reader module for a dump: `Kharma` for native Parthenon `.phdf` files,
`Iharm` otherwise. Dispatching per file rather than per run means a KHARMA
directory and an iharm3d directory both work, and KHARMA's iharm3d-format
output is correctly read by `Iharm`.
"""
reader_for(path::AbstractString) = Kharma.is_kharma_dump(String(path)) ? Kharma : Iharm

"""
    dump_times_of(paths)

Simulation time of each dump, read without loading any fluid primitives.
Delegates to `Utils.dump_time`, which already handles both the Parthenon `Info`
attribute and the iharm3d root `t` dataset.
"""
dump_times_of(paths::AbstractVector{<:AbstractString}) =
    Float64[Utils.dump_time(String(p)) for p in paths]

"""
    DumpSeries(paths; times = dump_times_of(paths))

A time-ordered dump series with a load cache.

The cache is keyed by dump index and is pruned to the index range a frame
needs, so the resident snapshot count is the brisk-light figure of merit rather
than an unbounded leak: bands whose intervals fall within the same dump spacing
share a single load, and frames that overlap in time pay nothing to re-enter a
dump they already hold.
"""
mutable struct DumpSeries{D}
    paths::Vector{String}
    times::Vector{Float64}
    cache::Dict{Int,D}
    n_loads::Int
    bytes_read::Int
    load_time::Float64
end

function DumpSeries(paths::AbstractVector{<:AbstractString};
                    times::AbstractVector{<:Real} = dump_times_of(paths),
                    snapshot_type = nothing)
    ps = String.(collect(paths))
    ts = Float64.(collect(times))
    length(ps) == length(ts) ||
        error("DumpSeries: $(length(ps)) paths but $(length(ts)) times")
    issorted(ts) || error("DumpSeries: dump times must be ascending")
    D = snapshot_type === nothing ? Any : snapshot_type
    return DumpSeries{D}(ps, ts, Dict{Int,D}(), 0, 0, 0.0)
end

Base.length(s::DumpSeries) = length(s.paths)

"""
    snapshot_bytes(data)

Approximate resident size of one snapshot: the sum of its array fields. Used
for the memory-footprint metric.
"""
function snapshot_bytes(data)
    total = 0
    for f in fieldnames(typeof(data))
        v = getfield(data, f)
        v isa AbstractArray && (total += sizeof(v))
    end
    return total
end

"""
    load_dump!(series, idx, Rhigh, model)

Load dump `idx`, reusing the cache when possible. Accounts the load so a run
can report dumps read, bytes read and time spent in I/O.
"""
function load_dump!(series::DumpSeries{D}, idx::Int, Rhigh, model) where {D}
    haskey(series.cache, idx) && return series.cache[idx]
    path = series.paths[idx]
    reader = reader_for(path)
    t0 = time()
    data = reader.load_data(path, Rhigh, model)
    series.load_time += time() - t0
    series.n_loads += 1
    series.bytes_read += snapshot_bytes(data)
    series.cache[idx] = data
    return data
end

"""
    prune_cache!(series, keep)

Drop every cached snapshot whose index is not in `keep`.
"""
function prune_cache!(series::DumpSeries, keep)
    for k in collect(keys(series.cache))
        k in keep || delete!(series.cache, k)
    end
    return series
end

"""
    frame_snapshots!(series, indices, Rhigh, model) -> SnapshotSet

Load the frame's dumps and pack them into a [`TimeInterp.SnapshotSet`](@ref),
pruning anything the frame does not need. The returned set's length is part of
its type, so the caller must cross a function barrier before the render loop —
see [`render_brisk_frame!`](@ref).
"""
function frame_snapshots!(series::DumpSeries, indices::Vector{Int}, Rhigh, model)
    snaps = [load_dump!(series, k, Rhigh, model) for k in indices]
    # Keep every cached dump between the frame's oldest and newest, not only the ones it reads: a lensed
    # band (n >= 1) reads the dumps the n = 0 band read 10-20 M earlier, so dropping them in the gap
    # between band windows made small-p runs load each dump once per band (p = 0: ~3 loads per dump).
    # Nothing is loaded into the gap; at p >= 0.5 the band windows already overlap and this keeps the same set.
    prune_cache!(series, minimum(indices):maximum(indices))
    return SnapshotSet(snaps, [series.times[k] for k in indices])
end


@inline step_band(::Nothing, off::Int, k::Int) = 0
@inline step_band(c::StepCache, off::Int, k::Int) = @inbounds Int(c.band[off + k])

@inline step_radiating(::Nothing, off::Int, k::Int, Xf, model, Rh) =
    Radiation.radiating_region(Xf, model, Rh)
@inline step_radiating(c::StepCache, off::Int, k::Int, Xf, model, Rh) =
    @inbounds c.radiating[off + k]

"""
    integrate_brisk!(traj, nsteps, Image, I, J, freq, bhspin, model, Rh,
                     cache, off, tm, snaps, bad_pixels)

Radiative transfer along one geodesic under a [`TimeInterp.TimeMap`](@ref).

Structurally this is `Radiation.integrate_emission!` with one change: before
each emissivity evaluation the step's coordinate time is replaced by the
absolute fluid time `map_time` gives it. Everything else — the far-end seed,
the `continue` on non-radiating steps that deliberately does *not* roll
`(ji, ki)` forward, the trapezoid in `approximate_solve`, the `dl` taken from
`traj[nstep-1]` — matches the fast-light path exactly, which is what lets the
`p = 0` run be compared to it bitwise.

The one addition is `pending`: when the band changes, the *leading* emissivity
is re-evaluated against the new band's time before the step is integrated.
Without it a single transfer step would average `j` and `alpha` drawn from two
different epochs, and it would do so at the equatorial plane, where the
emission is brightest. `pending` is cleared only once the re-evaluation has
actually happened, so a band change that falls inside a non-radiating gap does
not leave `(ji, ki)` stranded on the previous band.

Nothing is allocated: `traj[k].X` is an `SVector`, and the time-shifted
position is rebuilt as another `SVector` on the stack.
"""
function integrate_brisk!(traj, nsteps::Int, Image::AbstractMatrix{T},
                          I::Int, J::Int, freq, bhspin, model, Rh,
                          cache, off::Int,
                          tm::TimeMap, snaps::SnapshotSet,
                          bad_pixels::Threads.Atomic{Int};
                          tau_img = nothing) where {T}
    # tau_img: optical depth along the ray, the dtau = kavg * dl of approximate_solve (ipole's tau)

    if nsteps < 2
        @inbounds Image[I, J] = zero(T)
        tau_img === nothing || (@inbounds tau_img[I, J] = zero(T))
        return
    end

    b = step_band(cache, off, nsteps)
    @inbounds Xi = traj[nsteps].X
    @inbounds Kconi = traj[nsteps].Kcon
    Xi_abs = SVector{4,T}(map_time(tm, Xi[1], b), Xi[2], Xi[3], Xi[4])
    ji, ki = Radiation.get_jk(Xi_abs, Kconi, freq, bhspin, model, snaps)

    Intensity = zero(T)
    tau = zero(T)
    pending = false

    for nstep in nsteps:-1:2
        @inbounds Xf = traj[nstep - 1].X

        bn = step_band(cache, off, nstep)
        if bn != b
            b = bn
            pending = true
        end

        step_radiating(cache, off, nstep, Xf, model, Rh) || continue

        @inbounds Xi = traj[nstep].X
        @inbounds Kconi = traj[nstep].Kcon
        @inbounds Kconf = traj[nstep - 1].Kcon

        if pending
            Xi_abs = SVector{4,T}(map_time(tm, Xi[1], b), Xi[2], Xi[3], Xi[4])
            ji, ki = Radiation.get_jk(Xi_abs, Kconi, freq, bhspin, model, snaps)
            pending = false
        end

        Xf_abs = SVector{4,T}(map_time(tm, Xf[1], b), Xf[2], Xf[3], Xf[4])
        jf, kf = Radiation.get_jk(Xf_abs, Kconf, freq, bhspin, model, snaps)

        @inbounds Intensity = Radiation.approximate_solve(Intensity, ji, ki, jf, kf,
                                                          traj[nstep - 1].dl)
        @inbounds tau += (ki + kf) / 2 * traj[nstep - 1].dl

        if !isfinite(Intensity)
            Threads.atomic_add!(bad_pixels, 1)
            @inbounds Image[I, J] = T(NaN)
            tau_img === nothing || (@inbounds tau_img[I, J] = T(NaN))
            return
        end

        ji = jf
        ki = kf
    end

    @inbounds Image[I, J] = Intensity
    tau_img === nothing || (@inbounds tau_img[I, J] = tau)
    return
end

"""
    integrate_brisk_pol!(traj, nsteps, pol, I, J, freq, bhspin, model, Rh,
                         cache, off, tm, snaps, dl_unit, bad_pixels)

Polarized counterpart of [`integrate_brisk!`](@ref): the coherency tensor is
carried along the same geodesic with `Polarization.polarized_step`, exactly as
`Polarization.integrate_emission_pol` does for a single snapshot, and the one
change is the same as for Stokes I: each step's end point `Xf`, where the
transfer coefficients are evaluated, is moved to the absolute fluid time
`map_time` gives it for the step's band. Parallel transport uses the step's
start and half-step points, whose time coordinate does not enter (the Kerr
connection is stationary), so they are passed unshifted. The camera-frame
Stokes parameters and the Faraday depth go to `pol[:, I, J]` through
`Polarization.save_pixel!` (so they include `freq^3`, unlike `Image`).

As in ipole and `Polarization.PolSink`, this is a separate calculation from the
unpolarized `Image`: it uses the Dexter (2016) coefficients at the end of each
step, so its Stokes I differs slightly from `Image` by design.
"""
function integrate_brisk_pol!(traj, nsteps::Int, pol::AbstractArray{T,3},
                              I::Int, J::Int, freq, bhspin, model, Rh,
                              cache, off::Int,
                              tm::TimeMap, snaps::SnapshotSet, dl_unit,
                              bad_pixels::Threads.Atomic{Int}) where {T}
    if nsteps < 2
        Polarization.save_pixel!(pol, I, J, zero(T), zero(T), zero(T), zero(T), zero(T), freq, 0)
        return
    end

    N = Polarization.zero_tensor(T)
    tauF = zero(T)
    for nstep in nsteps:-1:2
        @inbounds ti = traj[nstep]
        @inbounds tf = traj[nstep - 1]
        b = step_band(cache, off, nstep)
        Xf = tf.X
        Xf_abs = SVector{4,T}(map_time(tm, Xf[1], b), Xf[2], Xf[3], Xf[4])
        N, tauF = Polarization.polarized_step(N, tauF, ti.X, ti.Kcon, ti.Xhalf, ti.Kconhalf,
                                              Xf_abs, tf.Kcon, tf.dl, dl_unit, Rh, freq,
                                              bhspin, model, snaps)
    end

    SI, SQ, SU, SV = Polarization.project_n(N, traj[1].X, bhspin, model)
    if !(isfinite(SI) && isfinite(SQ) && isfinite(SU) && isfinite(SV))
        Threads.atomic_add!(bad_pixels, 1)
        Polarization.save_pixel!(pol, I, J, T(NaN), T(NaN), T(NaN), T(NaN), tauF, freq, 0)
        return
    end
    Polarization.save_pixel!(pol, I, J, SI, SQ, SU, SV, tauF, freq, 0)
    return
end

"""
    render_brisk_frame!(Image, all_geodesics, nsteps, model, nx, ny, freq,
                        cache, tm, snaps; progress, desc)

Render one frame into `Image`, in place.

This exists as its own function to act as a type barrier: a `SnapshotSet`'s
length is part of its type and changes from frame to frame as `p` and the
geometry decide how many dumps a frame needs, so the dynamic dispatch is paid
once per frame here rather than once per pixel inside the loop. Julia then
specializes the whole transfer loop on both the snapshot count and the band
cache, which is what keeps `nearest_pair`'s bracket search unrolled.

`Image` is *not* scaled by `freq^3`; the caller does that, matching
`Imaging.raytrace_image`.
"""
function render_brisk_frame!(Image::AbstractMatrix{T},
                             all_geodesics, nsteps::AbstractMatrix{Int},
                             model, nx::Int, ny::Int, freq,
                             cache, tm::TimeMap, snaps::SnapshotSet;
                             progress::Bool = true,
                             desc::AbstractString = "brisk ",
                             pol::Union{Nothing,AbstractArray} = nothing,
                             tau::Union{Nothing,AbstractMatrix} = nothing) where {T}

    Rh = 1 + sqrt(1.0 - model.a * model.a)
    bad_pixels = Threads.Atomic{Int}(0)
    bad_pol = Threads.Atomic{Int}(0)
    dl_unit = model.L_unit * Constants.HPL / (Constants.ME * Constants.CL * Constants.CL)
    offsets = cache === nothing ? nothing : cache.offsets

    pbar = progress ? Progress(nx; desc = String(desc), showspeed = true, barlen = 30) : nothing
    plock = ReentrantLock()

    Threads.@threads :greedy for i in 1:nx
        for j in 1:ny
            off = offsets === nothing ? 0 : Int(@inbounds offsets[(j - 1) * nx + i])
            integrate_brisk!(all_geodesics[i, j], nsteps[i, j], Image, i, j,
                             freq, model.a, model, Rh, cache, off, tm, snaps,
                             bad_pixels; tau_img = tau)
            pol === nothing ||
                integrate_brisk_pol!(all_geodesics[i, j], nsteps[i, j], pol, i, j,
                                     freq, model.a, model, Rh, cache, off, tm, snaps,
                                     dl_unit, bad_pol)
        end
        if pbar !== nothing
            lock(plock) do
                ProgressMeter.next!(pbar)
            end
        end
    end
    pbar === nothing || finish!(pbar)

    bad_pixels[] > 0 && @warn "Brisk light: $(bad_pixels[]) pixel(s) gave a non-finite intensity and were set to NaN"
    bad_pol[] > 0 && @warn "Brisk light: $(bad_pol[]) pixel(s) gave non-finite Stokes parameters and were set to NaN"
    return bad_pixels[]
end

"""
    Imaging.raytrace_image_at_time!(Image, model, snaps, all_geodesics, nsteps,
                                    freq, t_frame; progress, desc)

Fast light at `t_frame`: a constant time map over `snaps`, so every pixel reads
the fluid at the same instant, interpolated between the two dumps bracketing it.
See the declaration in `Imaging` for why this lives here.
"""
function Imaging.raytrace_image_at_time!(Image::AbstractMatrix, model, snaps::SnapshotSet,
                                         all_geodesics, nsteps::AbstractMatrix{Int},
                                         freq, t_frame;
                                         progress::Bool = true,
                                         desc::AbstractString = "fast light ",
                                         pol::Union{Nothing,AbstractArray} = nothing,
                                         tau::Union{Nothing,AbstractMatrix} = nothing)
    nx, ny = size(Image)
    ts = snapshot_times(snaps)
    tm = fastlight_timemap(float(t_frame), float(ts[1]), float(ts[end]))
    return render_brisk_frame!(Image, all_geodesics, nsteps, model, nx, ny, freq,
                               nothing, tm, snaps; progress = progress, desc = desc, pol = pol, tau = tau)
end

"""
    process_fastlight_images!(series, all_geodesics, nsteps, model, nx, ny, freq,
                              Rhigh; kw...)

Render a fast-light movie on an observer-time cadence, interpolating the fluid
to each frame time.

This is the `n_bands = 0`, `p = 0` degenerate case of brisk light, and it exists
so that a fast-light run can be put on exactly the same frame grid as a brisk or
slow-light run. Each frame loads the one or two dumps bracketing its time, which
is also the reference point every brisk-light speed-up is measured against.

Keywords match [`process_brisklight_images!`](@ref) where they overlap.
"""
function process_fastlight_images!(series::DumpSeries,
                                   all_geodesics, nsteps::AbstractMatrix{Int},
                                   model, nx::Int, ny::Int, freq, Rhigh;
                                   cadence::Real = 10.0,
                                   t_start::Union{Nothing,Real} = nothing,
                                   n_frames::Union{Nothing,Integer} = nothing,
                                   snap_to_dump::Bool = true,
                                   t_frames::Union{Nothing,AbstractVector} = nothing,
                                   output_dir::Union{Nothing,AbstractString} = nothing,
                                   output_prefix::AbstractString = "fast",
                                   output_format::AbstractString = "ipole",
                                   meta::Union{Nothing,Dict{String,Any}} = nothing,
                                   progress::Bool = true,
                                   verbose::Bool = true,
                                   polarized::Bool = false,
                                   t_ref::Real = NaN)

    # `t_ref`: band-0 modal emission time of the same geometry (BandStats.t_ref, negative), when known.
    # Fast light renders the whole flow at t_frame, which is the n = 0 clock; with t_ref the camera time
    # of the matching slow-light frame, t_frame - t_ref, is recorded too. NaN: not known, recorded as NaN.
    dt = series.times
    sched = if t_frames !== nothing
        Float64.(collect(t_frames))
    else
        t0 = t_start === nothing ? dt[1] : Float64(t_start)
        if t_start === nothing && snap_to_dump
            t0 = dt[1]
        end
        times = Float64[]
        t = t0
        while t <= dt[end] + 1e-9
            push!(times, t)
            n_frames !== nothing && length(times) >= n_frames && break
            t += Float64(cadence)
        end
        times
    end

    outdir = output_dir === nothing ? nothing : String(output_dir)
    outdir === nothing || mkpath(outdir)
    manifest = outdir === nothing ? nothing : joinpath(outdir, "frames.csv")
    if manifest !== nothing
        open(manifest, "w") do io
            println(io, "index,t_frame,t_camera,file,n_snapshots,dumps_per_band,",
                        "n_loads,bytes_read,load_seconds,render_seconds,clamped_fraction,resident_bytes")
        end
    end

    Image = zeros(Float64, nx, ny)
    Tau = zeros(Float64, nx, ny)
    pol = polarized ? zeros(Float64, Polarization.NIMG, nx, ny) : nothing
    rows = NamedTuple[]
    # Band-0-only statistics, so the manifest columns line up with a brisk run
    # and the two can be diffed directly.
    dummy = BandStats{Float64}(0, [0.0], [0.0], [0.0], [0.0], [0.0],
                               [true], [0], [NaN], 0.0, 0.0, (0.0, 0.0))

    for (k, t_frame) in enumerate(sched)
        idx = collect(band_dump_range(t_frame, t_frame, dt))
        loads0, bytes0, lt0 = series.n_loads, series.bytes_read, series.load_time
        snaps = frame_snapshots!(series, idx, Rhigh, model)
        n_loads = series.n_loads - loads0
        bytes_read = series.bytes_read - bytes0
        load_seconds = series.load_time - lt0

        fill!(Image, 0.0)
        pol === nothing || fill!(pol, 0.0)
        render_seconds = @elapsed Imaging.raytrace_image_at_time!(
            Image, model, snaps, all_geodesics, nsteps, freq, t_frame;
            progress = progress, desc = @sprintf("fast t=%.1fM ", t_frame), pol = pol, tau = Tau)
        Image .*= freq^3

        resident = sum(snapshot_bytes(v) for v in values(series.cache); init = 0)
        file = ""
        if outdir !== nothing
            file = write_frame(outdir, output_prefix, output_format, k, t_frame,
                               Image, snaps, dummy, meta; pol = pol, t_camera = t_frame - t_ref, tau = Tau)
            open(manifest, "a") do io
                @printf(io, "%d,%.10f,%.10f,%s,%d,%d,%d,%d,%.4f,%.4f,%.6f,%d\n",
                        k, t_frame, t_frame - t_ref, basename(file), length(idx), length(idx),
                        n_loads, bytes_read, load_seconds, render_seconds, 0.0, resident)
            end
        end

        push!(rows, (index = k, t_frame = t_frame, t_camera = t_frame - t_ref, file = file,
                     dump_indices = idx, n_snapshots = length(idx),
                     dumps_per_band = [length(idx)], n_loads = n_loads,
                     bytes_read = bytes_read, load_seconds = load_seconds,
                     render_seconds = render_seconds, clamped_fraction = 0.0,
                     resident_bytes = resident))

        verbose && @info @sprintf("  fast frame %d/%d  t = %.3f M  %d snapshot(s)  render %.2f s",
                                  k, length(sched), t_frame, length(idx), render_seconds)
    end

    verbose && !isempty(rows) && @info @sprintf("Fast light: %d frame(s), %.2f snapshots/image, %d load(s), %.2f GB read",
        length(rows), sum(r.n_snapshots for r in rows) / length(rows),
        series.n_loads, series.bytes_read / 2^30)

    return (frames = length(rows), metrics = rows, schedule = sched)
end

# ---------------------------------------------------------------------------
# Frame schedule
# ---------------------------------------------------------------------------

"""
    frame_window(stats, dump_times) -> (t_first, t_last)

Range of frame times the loaded series can serve without freezing the plasma:
the earliest and latest `t_frame` for which every time the retained support
needs still falls inside `[dump_times[1], dump_times[end]]`.

Because the support shrinks with `p`, this window *widens* as `p` falls — a
low-`p` run covers a longer observer baseline from the same dumps than slow
light can, which is a result worth reporting rather than a coincidence.
"""
function frame_window(s::BandStats{T}, dump_times::AbstractVector) where {T}
    t_lo, t_hi = retained_support(s)
    return (T(dump_times[1]) - t_lo + s.t_ref, T(dump_times[end]) - t_hi + s.t_ref)
end

"""
    frame_schedule(stats, dump_times; cadence, t_start, n_frames, snap_to_dump)

Vector of frame times.

With `snap_to_dump` (the default) the first frame is moved forward onto the
nearest dump time at or after the start of the available window. Advancing from
there by a cadence that is a multiple of the dump spacing keeps every frame's
`n = 0` band exactly on a dump, so each frame has a fast-light counterpart that
uses precisely the same file. A cadence finer than the dump spacing is allowed
and simply interpolates, for `n = 0` as for every other band.
"""
function frame_schedule(s::BandStats{T}, dump_times::AbstractVector;
                        cadence::Real = 10.0,
                        t_start::Union{Nothing,Real} = nothing,
                        n_frames::Union{Nothing,Integer} = nothing,
                        snap_to_dump::Bool = true) where {T}

    t_first, t_last = frame_window(s, dump_times)
    t0 = t_start === nothing ? t_first : T(t_start)

    if t_start === nothing && snap_to_dump
        k = searchsortedfirst(dump_times, t_first)
        # Only snap if the dump we land on is still inside the usable window.
        # When the retained support nearly fills the series that window can be
        # narrower than one dump spacing, and snapping forward would push the
        # first frame past the end -- losing every frame rather than giving up
        # the (cosmetic) alignment.
        if k <= length(dump_times) && T(dump_times[k]) <= t_last
            t0 = T(dump_times[k])
        elseif k <= length(dump_times)
            @info @sprintf("Brisk light: the usable frame window [%.3f, %.3f] M is narrower than the dump spacing, so the first frame is not aligned to a dump; the n = 0 band will interpolate between two snapshots instead of reading one.",
                           t_first, t_last)
        end
    end

    dt = T(cadence)
    dt > 0 || error("frame_schedule: cadence must be positive, got $cadence")

    times = T[]
    t = t0
    while t <= t_last + eps(T(1)) * max(one(T), abs(t_last)) * 8
        push!(times, t)
        n_frames !== nothing && length(times) >= n_frames && break
        t += dt
    end

    if isempty(times)
        @warn @sprintf("Brisk light: no frame fits. Available window for p = %.3f is [%.3f, %.3f] M and t_start = %.3f.",
                       s.p, t_first, t_last, t0)
    end
    return times
end

# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

"""
    process_brisklight_images!(stats, cache, series, all_geodesics, nsteps, model,
                               nx, ny, freq, Rhigh; kw...)

Render a brisk-light movie and return a metrics table.

Geodesics, the band walk and the delay sample are all inputs, so a `p` sweep
pays for them once: rebuild `stats` with [`band_stats`](@ref) at the new `p`
and call this again with the same `cache`.

Each frame loads only the dumps its bands' retained intervals bracket, prunes
everything else, and records what it cost. The returned table is what the
performance assessment is built from.

# Keywords
- `cadence`, `t_start`, `n_frames`, `snap_to_dump`, `t_frames`: the schedule
  (see [`frame_schedule`](@ref)). `t_frames` overrides the rest.
- `sample`: the [`BandSample`](@ref) from [`band_walk`](@ref). Supply it to get
  the clamped-fraction diagnostic, which is the gate on whether a run is a
  valid brisk-light run at all.
- `output_dir`, `output_prefix`, `output_format`, `meta`: output. `meta` is the
  camera/observer dictionary `Output.generate_output_file` needs; when it is
  `nothing` only the raw image and the manifest are written.
- `clamp_warn`: warn above this clamped fraction.
- `interpolate_bands`: interpolate between the loaded snapshots instead of
  clipping each step into its band's interval (default `true`); see
  [`build_timemap`](@ref).
"""
function process_brisklight_images!(stats::BandStats{T},
                                    cache,
                                    series::DumpSeries,
                                    all_geodesics,
                                    nsteps::AbstractMatrix{Int},
                                    model,
                                    nx::Int, ny::Int,
                                    freq, Rhigh;
                                    cadence::Real = 10.0,
                                    t_start::Union{Nothing,Real} = nothing,
                                    n_frames::Union{Nothing,Integer} = nothing,
                                    snap_to_dump::Bool = true,
                                    t_frames::Union{Nothing,AbstractVector} = nothing,
                                    sample::Union{Nothing,BandSample{T}} = nothing,
                                    output_dir::Union{Nothing,AbstractString} = nothing,
                                    output_prefix::AbstractString = "brisk",
                                    output_format::AbstractString = "ipole",
                                    meta::Union{Nothing,Dict{String,Any}} = nothing,
                                    clamp_warn::Real = 0.01,
                                    progress::Bool = true,
                                    verbose::Bool = true,
                                    polarized::Bool = false,
                                    skip_existing::Bool = false,
                                    interpolate_bands::Bool = true) where {T}
    # skip_existing: frames whose output file is already in output_dir are not rendered again, so a run cut
    # short (a 1 h interactive session, a preempted job) picks up where it stopped. Frame k keeps its index, as
    # the schedule depends only on t_start, cadence and n_frames; files are renamed into place once written,
    # so one that exists is complete.

    dump_times = series.times
    length(dump_times) >= 1 || error("process_brisklight_images!: empty dump series")

    sched = t_frames === nothing ?
            frame_schedule(stats, dump_times; cadence = cadence, t_start = t_start,
                           n_frames = n_frames, snap_to_dump = snap_to_dump) :
            T.(collect(t_frames))

    t_lo, t_hi = retained_support(stats)
    t_first, t_last = frame_window(stats, dump_times)
    if verbose
        @info @sprintf("Brisk light: p = %.3f  retained support = [%.4f, %.4f] M (width %.4f M)",
                       stats.p, t_lo, t_hi, t_hi - t_lo)
        @info @sprintf("Brisk light: usable frame window = [%.4f, %.4f] M; rendering %d frame(s)",
                       t_first, t_last, length(sched))
    end
    isempty(sched) && return (frames = 0, metrics = NamedTuple[], stats = stats)

    outdir = output_dir === nothing ? nothing : String(output_dir)
    outdir === nothing || mkpath(outdir)
    manifest = outdir === nothing ? nothing : joinpath(outdir, "frames.csv")
    if manifest !== nothing && !(skip_existing && isfile(manifest))
        open(manifest, "w") do io
            println(io, "index,t_frame,t_camera,file,n_snapshots,dumps_per_band,",
                        "n_loads,bytes_read,load_seconds,render_seconds,clamped_fraction,resident_bytes")
        end
    end

    Image = zeros(T, nx, ny)
    Tau = zeros(T, nx, ny)
    pol = polarized ? zeros(T, Polarization.NIMG, nx, ny) : nothing
    rows = NamedTuple[]

    n_skipped = 0
    for (k, t_frame) in enumerate(sched)
        if skip_existing && outdir !== nothing &&
           isfile(joinpath(outdir, @sprintf("%s.%05d.%s", output_prefix, k, meta === nothing ? "txt" : "h5")))
            n_skipped += 1
            continue
        end
        idx, per_band = frame_dump_indices(stats, t_frame, dump_times)

        loads0 = series.n_loads
        bytes0 = series.bytes_read
        ltime0 = series.load_time
        snaps = frame_snapshots!(series, idx, Rhigh, model)
        n_loads = series.n_loads - loads0
        bytes_read = series.bytes_read - bytes0
        load_seconds = series.load_time - ltime0

        tA = dump_times[first(idx)]
        tB = dump_times[last(idx)]
        tm = build_timemap(stats, t_frame, tA, tB; interpolate_bands = interpolate_bands)

        clamped = T(NaN)
        if sample !== nothing
            clamped, per_bin = clamped_fraction(stats, sample, t_frame, tA, tB)
            if isfinite(clamped) && clamped > clamp_warn
                @warn @sprintf("Brisk light: frame t_frame = %.4f M has %.2f%% of steps on the availability clamp (frozen plasma); this run is not comparable to slow light. Per band: %s",
                               t_frame, 100 * clamped,
                               join([@sprintf("n=%d:%.1f%%", b - 1, 100 * per_bin[b])
                                     for b in eachindex(per_bin) if isfinite(per_bin[b])], " "))
            end
        end

        fill!(Image, zero(T))
        pol === nothing || fill!(pol, zero(T))
        render_seconds = @elapsed render_brisk_frame!(
            Image, all_geodesics, nsteps, model, nx, ny, freq, cache, tm, snaps;
            progress = progress,
            desc = @sprintf("brisk p=%.2f t=%.1fM ", stats.p, t_frame), pol = pol, tau = Tau)
        Image .*= freq^3

        resident = sum(snapshot_bytes(v) for v in values(series.cache); init = 0)
        dumps_per_band = [length(r) for r in per_band]

        file = ""
        if outdir !== nothing
            file = write_frame(outdir, output_prefix, output_format, k, t_frame,
                               Image, snaps, stats, meta; pol = pol, t_camera = t_frame - stats.t_ref, tau = Tau)
            open(manifest, "a") do io
                @printf(io, "%d,%.10f,%.10f,%s,%d,%s,%d,%d,%.4f,%.4f,%.6f,%d\n",
                        k, t_frame, t_frame - stats.t_ref, basename(file),
                        length(idx), join(dumps_per_band, "|"),
                        n_loads, bytes_read, load_seconds, render_seconds,
                        isfinite(clamped) ? clamped : NaN, resident)
            end
        end

        push!(rows, (index = k, t_frame = t_frame, t_camera = t_frame - stats.t_ref,
                     file = file, dump_indices = idx, n_snapshots = length(idx),
                     dumps_per_band = dumps_per_band, n_loads = n_loads,
                     bytes_read = bytes_read, load_seconds = load_seconds,
                     render_seconds = render_seconds, clamped_fraction = clamped,
                     resident_bytes = resident))

        verbose && @info @sprintf("  frame %d/%d  t_frame = %.3f M  %d snapshot(s) [%s]  %d load(s)  render %.2f s  clamped %.3f%%",
                                  k, length(sched), t_frame, length(idx),
                                  join(dumps_per_band, "|"), n_loads, render_seconds,
                                  isfinite(clamped) ? 100 * clamped : NaN)
    end

    verbose && n_skipped > 0 && @info "Brisk light: $n_skipped frame(s) already in $outdir, skipped"
    if verbose && !isempty(rows)
        tot_render = sum(r.render_seconds for r in rows)
        tot_load = sum(r.load_seconds for r in rows)
        mean_snaps = sum(r.n_snapshots for r in rows) / length(rows)
        @info @sprintf("Brisk light: %d frame(s), p = %.3f, %.2f snapshots/image, %d total load(s), %.2f GB read, render %.2f s, I/O %.2f s",
                       length(rows), stats.p, mean_snaps, series.n_loads,
                       series.bytes_read / 2^30, tot_render, tot_load)
        manifest === nothing || @info "Brisk light: manifest -> $manifest"
    end

    return (frames = length(rows), metrics = rows, stats = stats,
            schedule = sched, retained_support = (t_lo, t_hi),
            frame_window = (t_first, t_last))
end

"""
    write_frame(dir, prefix, format, index, t_frame, Image, snaps, stats, meta)

Write one frame. With `meta` supplied the frame goes out through
`Output.generate_output_file` in ipole's HDF5 layout, stamped with
`img_time = t_frame` (the `n = 0` clock) so downstream tools can pair frames by
time; otherwise it is written as plain text, which is enough for the
comparison tests.
"""
function write_frame(dir::AbstractString, prefix::AbstractString,
                     format::AbstractString, index::Int, t_frame,
                     Image, snaps::SnapshotSet, stats::BandStats,
                     meta::Union{Nothing,Dict{String,Any}};
                     pol::Union{Nothing,AbstractArray} = nothing,
                     t_camera::Real = NaN,
                     tau::Union{Nothing,AbstractMatrix} = nothing)
    if meta === nothing
        path = joinpath(dir, @sprintf("%s.%05d.txt", prefix, index))
        open(path, "w") do io
            for i in axes(Image, 1)
                println(io, join(view(Image, i, :), '\t'))
            end
        end
        return path
    end

    path = joinpath(dir, @sprintf("%s.%05d.h5", prefix, index))
    data = copy(meta)
    data["image"] = Image
    data["img_time"] = t_frame
    data["data"] = snaps.snaps[1]
    pol === nothing || (data["pol"] = pol)
    tau === nothing || (data["tau"] = tau)
    data["frame_index"] = index
    data["t_fluid"] = t_frame
    data["t_camera"] = t_camera
    # written under a temporary name and renamed, so a file at `path` is always complete (see skip_existing)
    tmp = joinpath(dir, @sprintf("%s.%05d.partial.h5", prefix, index))
    Output.generate_output_file(tmp, data; format = String(format))
    mv(tmp, path; force = true)
    return path
end

end # module Brisklight
