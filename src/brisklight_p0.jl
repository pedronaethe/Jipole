"""
Brisk-light rendering at p = 0: every observer frame uses a single GRMHD
snapshot per lensing band, taken at the band's modal emission time.
"""
module Brisklight_p0

using HDF5
using Printf
using Dates
using ProgressMeter
using Statistics
using KernelDensity
using ..Constants
using ..Coordinates
using ..Radiation
using ..Iharm
using ..Output

export OfBriskLight, get_dump_time, modal_hdi_kde, band_geometry,
    compute_band_modal_times!, interpolation_process, process_brisklight_images!

# Bin layout, same convention as feature/brisk-light:
#   1 .. n_bands+1 -> lensing band n = 0 .. n_bands (higher bands are merged into the last one)
#   n_bands+2      -> approach: steps traversed before the first midplane crossing
#   n_bands+3      -> shadow: every step of a ray that never crosses the midplane
@inline nbins(n_bands::Int) = n_bands + 3
@inline approach_bin_index(n_bands::Int) = n_bands + 2
@inline shadow_bin_index(n_bands::Int) = n_bands + 3

@inline function bin_index(band::Int, seed::Int, n_bands::Int)
    seed == 0 && return shadow_bin_index(n_bands)
    band == seed && return approach_bin_index(n_bands)
    return min(band, n_bands) + 1
end

function bin_name(b::Int, n_bands::Int)
    b == approach_bin_index(n_bands) && return "approach"
    b == shadow_bin_index(n_bands) && return "shadow"
    return "n=$(b - 1)"
end


"""
    OfBriskLight(n_bands, image_cadence; p = 0.0)

Brisk-light run state. `n_bands` is declared in the notebook, like the spin or
the black-hole mass: it is the highest lensing band kept as its own bin, and the
number of bands actually resolved depends on the image resolution. The per-bin
statistics are filled by [`compute_band_modal_times!`](@ref).
"""
mutable struct OfBriskLight
    n_bands::Int
    p::Float64
    image_cadence::Float64
    modal_times::Vector{Float64}
    hdi_intervals::Vector{NTuple{2,Float64}}
end

OfBriskLight(n_bands::Int, image_cadence::Real; p::Real = 0.0) =
    OfBriskLight(n_bands, Float64(p), Float64(image_cadence),
                 fill(NaN, nbins(n_bands)), fill((NaN, NaN), nbins(n_bands)))


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
    band_geometry(midplane_crossings, all_geodesics, nsteps, pixels_x, pixels_y,
                  n_bands, model; pixel_stride = 4)

Walk every geodesic once, from its far end toward the camera, and assign each
transfer step `traj[nstep] -> traj[nstep - 1]` to a bin. The running band
counter starts at `midplane_crossings[i, j]` and drops by one at every midplane
crossing, as in `feature/brisk-light` (`collect_band_step_times` and
`precompute_step_info`). This is the only place where `bl_coord` is called.

Returns `(band_ts, step_bin, used)`:
- `band_ts[b]`: step times `X[1]` of bin `b`, sampled on every `pixel_stride`-th
  pixel along each axis. This is the KDE sample.
- `step_bin[i, j][nstep]`: bin of step `nstep -> nstep - 1`, for every pixel.
- `used[b]`: whether any pixel of the image has a step in bin `b`.
"""
function band_geometry(midplane_crossings::Matrix{Int}, all_geodesics, nsteps::Matrix{Int},
                       pixels_x::Int, pixels_y::Int, n_bands::Int, model;
                       pixel_stride::Int = 4)
    nb = nbins(n_bands)
    nt = Threads.maxthreadid()
    ts_thread = [[Float64[] for _ in 1:nb] for _ in 1:nt]
    used_thread = [falses(nb) for _ in 1:nt]
    step_bin = Matrix{Vector{Int8}}(undef, pixels_x, pixels_y)

    Threads.@threads :static for i in 1:pixels_x
        tid = Threads.threadid()
        for j in 1:pixels_y
            n = nsteps[i, j]
            bins = zeros(Int8, n)
            step_bin[i, j] = bins
            n < 2 && continue

            traj = all_geodesics[i, j]
            sampled = (i - 1) % pixel_stride == 0 && (j - 1) % pixel_stride == 0
            seed = midplane_crossings[i, j]
            band = seed
            above_prev = Coordinates.bl_coord(traj[n].X, model)[2] < π / 2

            for nstep in n:-1:2
                Xf = traj[nstep - 1].X
                above = Coordinates.bl_coord(Xf, model)[2] < π / 2
                if above != above_prev
                    band = max(band - 1, 0)
                    above_prev = above
                end
                b = bin_index(band, seed, n_bands)
                bins[nstep] = b
                used_thread[tid][b] = true
                sampled && push!(ts_thread[tid][b], Xf[1])
            end
        end
    end

    max_crossings = maximum(midplane_crossings)
    max_crossings - 1 > n_bands &&
        @info "Rays cross the midplane up to $max_crossings times: bands above n = $n_bands are merged into bin n=$n_bands."

    band_ts = [reduce(vcat, (ts_thread[t][b] for t in 1:nt)) for b in 1:nb]
    used = reduce((a, b) -> a .| b, used_thread)
    return band_ts, step_bin, used
end


"""
    compute_band_modal_times!(params_brisklight, band_ts; kde_kwargs...)

Fill `modal_times` and `hdi_intervals` with one [`modal_hdi_kde`](@ref) per bin,
at `params_brisklight.p`. Every bin with samples goes through the same KDE,
whatever its number of pixels; bins without samples are left as `NaN`. Valid for
any p: only the rendering is restricted to p = 0.
"""
function compute_band_modal_times!(params_brisklight::OfBriskLight, band_ts::Vector{Vector{Float64}};
                                   kde_kwargs...)
    n_bands = params_brisklight.n_bands
    length(band_ts) == nbins(n_bands) || error("Expected $(nbins(n_bands)) bins, got $(length(band_ts)).")

    for b in eachindex(band_ts)
        ts = band_ts[b]
        if isempty(ts)
            params_brisklight.modal_times[b] = NaN
            params_brisklight.hdi_intervals[b] = (NaN, NaN)
            continue
        end
        res = modal_hdi_kde(ts, params_brisklight.p; kde_kwargs...)
        params_brisklight.modal_times[b] = res.mode
        params_brisklight.hdi_intervals[b] = res.interval
        @info @sprintf("Brisk-light bin %-8s (p = %.3f): t_modal = %10.3f M  HDI = [%.3f, %.3f] M  (%d samples)",
                       bin_name(b, n_bands), params_brisklight.p, res.mode,
                       res.interval[1], res.interval[2], length(ts))
    end
    return params_brisklight
end


"""
    interpolation_process(t_e, dump_times, load!, buffer; interpolation = true)

Return the GRMHD snapshot at source time `t_e`, together with the range of
indices (into `dump_times`) of the dumps it was built from.

- `interpolation = true`: linear interpolation between the bracketing dumps
  `k` and `k + 1`, with the same weight as slow light (`Iharm.set_tinterp_ns`):
  `tinterp = 1 - (t_e - tA) / (tB - tA)`, value `tinterp * A + (1 - tinterp) * B`.
  Slow light applies this weight to every field after a linear spatial
  interpolation (`Grid.interp_scalar_time`); both operations are linear, so
  blending the full arrays first gives the same emissivities. The blend is
  written into `buffer`, a snapshot preallocated by the caller.
- `interpolation = false`: the nearest dump, returned as loaded (`buffer` is
  not used).

`load!(k)` must return dump `k`, reading it from disk only the first time.
"""
function interpolation_process(t_e::Float64, dump_times::Vector{Float64}, load!, buffer;
                               interpolation::Bool = true)
    k = clamp(searchsortedlast(dump_times, t_e), 1, length(dump_times) - 1)
    tA, tB = dump_times[k], dump_times[k + 1]

    if !interpolation
        kn = t_e - tA <= tB - t_e ? k : k + 1
        return load!(kn), kn:kn
    end

    A, B = load!(k), load!(k + 1)
    tinterp = 1.0 - (t_e - tA) / (tB - tA)
    fields = Base.tail(fieldnames(typeof(buffer)))  # every field except the time `t`
    for f in fields
        dst, a, b = getfield(buffer, f), getfield(A, f), getfield(B, f)
        @. dst = tinterp * a + (1.0 - tinterp) * b
    end
    return typeof(buffer)(t_e, (getfield(buffer, f) for f in fields)...), k:(k + 1)
end


"""
    integrate_pixel(traj, n, bins, band_data, freq, model)

Radiative transfer along one geodesic, from its far end `traj[n]` toward the
camera, as in slow light but with static snapshots: step `nstep -> nstep - 1`
uses `band_data[bins[nstep]]`. When the bin changes, the leading emissivity is
re-evaluated with the new snapshot, so a single step never mixes two snapshots.
"""
function integrate_pixel(traj, n::Int, bins::Vector{Int8}, band_data, freq, model)
    n < 3 && return 0.0
    b = bins[n]
    ji, ki = Radiation.get_jk(traj[n].X, traj[n].Kcon, freq, model.a, model, band_data[b])
    Intensity = 0.0

    # Stops at the step 3 -> 2, like slow light (`while nstep > 2`).
    for nstep in n:-1:3
        if bins[nstep] != b
            b = bins[nstep]
            ji, ki = Radiation.get_jk(traj[nstep].X, traj[nstep].Kcon, freq, model.a, model, band_data[b])
        end
        jf, kf = Radiation.get_jk(traj[nstep - 1].X, traj[nstep - 1].Kcon, freq, model.a, model, band_data[b])
        Intensity = Radiation.approximate_solve(Intensity, ji, ki, jf, kf, traj[nstep - 1].dl)
        ji, ki = jf, kf
    end
    return Intensity
end


"""
    render_frame_cpu!(Image, all_geodesics, nsteps, step_bin, band_data, pixels_x, pixels_y, freq, model, t_obs)

Integrate every pixel of one frame, threaded over pixels.
"""
function render_frame_cpu!(Image, all_geodesics, nsteps, step_bin, band_data,
                           pixels_x, pixels_y, freq, model, t_obs)
    p = Progress(pixels_x * pixels_y; desc = @sprintf("Rendering t_obs = %.1f M...", t_obs),
                 showspeed = true, barlen = 30)
    progress_lock = ReentrantLock()

    Threads.@threads :greedy for i in 1:pixels_x
        for j in 1:pixels_y
            Image[i, j] = integrate_pixel(all_geodesics[i, j], nsteps[i, j], step_bin[i, j],
                                          band_data, freq, model)
        end
        lock(progress_lock) do
            ProgressMeter.next!(p; step = pixels_y)
        end
    end
    finish!(p)
    return Image
end


"""
    process_brisklight_images!(params_brisklight, all_geodesics, nsteps, step_bin, used,
        model, pixels_x, pixels_y, freq, Rhigh, all_dumps_path, dump_indices,
        Xcamera, ro, theta_o, phi, fovx, fovy, SourceD, scale;
        interpolation = true, t_obs_start = nothing)

Render a p = 0 brisk-light movie. For each observer time `t_obs`, spaced by
`params_brisklight.image_cadence`, every bin `b` present in the image receives a
single snapshot at `t_obs + modal_times[b]` from [`interpolation_process`](@ref),
and every geodesic is integrated once against those snapshots. Each frame is
independent, so no image has to stay open across dump loads as in slow light.

The observer window is the widest one for which every bin's source time lies
inside the dump sequence; `t_obs_start` can move its start later (e.g. to align
the frames with a slow-light run). Each dump is read from disk at most once.

# Arguments
- `params_brisklight`: Run state, with `modal_times` already filled and `p = 0`.
- `all_geodesics`, `nsteps`: Traced geodesics and their lengths, one per pixel.
- `step_bin`, `used`: From [`band_geometry`](@ref).
- `model`: Iharm model parameters, built with `slow_light = false`.
- `all_dumps_path`: `Printf`-style format string for the dump sequence.
- `dump_indices`: File indices of the dumps available, in time order.
- `Xcamera`, `ro`, `theta_o`, `phi`, `fovx`, `fovy`, `SourceD`, `scale`, `Rhigh`:
  Written to the output file, as in slow light.
"""
function process_brisklight_images!(
    params_brisklight::OfBriskLight, all_geodesics, nsteps, step_bin, used,
    model, pixels_x, pixels_y, freq, Rhigh, all_dumps_path, dump_indices,
    Xcamera, ro, theta_o, phi, fovx, fovy, SourceD, scale;
    interpolation::Bool = true, t_obs_start::Union{Nothing,Float64} = nothing
)
    params_brisklight.p == 0.0 || error("This renderer only supports p = 0.")
    model.slow_light && error("Build the model with slow_light = false: brisk p = 0 uses static snapshots.")

    n_bands = params_brisklight.n_bands
    bins = findall(used)
    t_modal = params_brisklight.modal_times[bins]
    all(isfinite, t_modal) ||
        error("A bin present in the image has no KDE sample: rerun band_geometry with pixel_stride = 1.")

    dump_times = [get_dump_time(d, all_dumps_path) for d in dump_indices]
    issorted(dump_times) || error("Dump times are not increasing.")

    t_first = dump_times[1] - minimum(t_modal)
    t_last = dump_times[end] - maximum(t_modal)
    if t_obs_start !== nothing
        t_obs_start >= t_first || error("t_obs_start must be >= $t_first M.")
        t_first = t_obs_start
    end
    frames = t_first:params_brisklight.image_cadence:t_last
    isempty(frames) && error("Empty observer window: the dumps span less than the delay between bins.")
    @info @sprintf("Brisk-light p = 0: %d frames, t_obs = %.2f : %.2f : %.2f M, interpolation = %s",
                   length(frames), first(frames), params_brisklight.image_cadence, last(frames), interpolation)

    # Dump cache. For every bin, t_obs + t_modal grows from frame to frame, so the
    # dump indices it needs never decrease. A dump below the smallest index used in
    # the current frame is therefore never needed again and is dropped; any other
    # dump stays in `cache`, so no dump is ever read twice.
    cache = Dict{Int,Any}()
    nloads = Ref(0)
    load!(k) = get!(cache, k) do
        nloads[] += 1
        Iharm.load_data(Printf.format(Printf.Format(all_dumps_path), dump_indices[k]), Rhigh, model)
    end

    buffers = Vector{Any}(nothing, length(bins))
    if interpolation
        k0 = clamp(searchsortedlast(dump_times, first(frames) + t_modal[1]), 1, length(dump_times) - 1)
        buffers = [deepcopy(load!(k0)) for _ in bins]
    end

    position = Dict(b => m for (m, b) in enumerate(bins))
    output_dir = joinpath("..", "brisk_sims", Dates.format(now(), "yyyy-mm-dd-HH:MM:SS"))
    mkpath(output_dir)
    println("Outputs will be saved to: $output_dir")
    Image = zeros(Float64, pixels_x, pixels_y)

    for t_obs in frames
        out = [interpolation_process(t_obs + t_modal[m], dump_times, load!, buffers[m];
                                     interpolation = interpolation) for m in eachindex(bins)]
        snaps = first.(out)
        band_data = [[snaps[get(position, b, 1)]] for b in 1:nbins(n_bands)]

        kmin = minimum(first(last(o)) for o in out)
        filter!(kv -> kv.first >= kmin, cache)

        @info @sprintf("t_obs = %.2f M | ", t_obs) *
              join(["$(bin_name(b, n_bands)): $(dump_indices[last(out[m])])" for (m, b) in enumerate(bins)], "  ") *
              " | dumps in memory: $(length(cache))"

        render_frame_cpu!(Image, all_geodesics, nsteps, step_bin, band_data, pixels_x, pixels_y, freq, model, t_obs)

        file_name = joinpath(output_dir, Printf.format(Printf.Format("Image.%07.1f.h5"), t_obs))
        out_data = Dict{String,Any}(
            "image"    => Image .* freq^3,
            "img_time" => t_obs,
            "params"   => model,
            "data"     => band_data[1][1],
            "ro"       => ro,
            "theta_o"  => theta_o,
            "phi"      => phi,
            "fovx"     => fovx,
            "fovy"     => fovy,
            "freq"     => freq,
            "SourceD"  => SourceD,
            "scale"    => scale,
            "Xcamera"  => Xcamera,
            "Rhigh"    => Rhigh
        )
        Output.generate_output_file(file_name, out_data; format = "ipole")
        println("Saving image $(file_name)")
    end

    @info "Brisk-light p = 0: $(length(frames)) frames, $(nloads[]) dumps read from disk (each at most once)."
    return frames
end

end
