"""
Brisk light (Rojas-Paternina & Cardenas-Avendano 2026), and fast light on a cadence: movies
from a dump series over rays traced once, where each frame reads only the dumps it needs.

Everything else is Jipole's own: the rays come from `Slowlight.trace_slowlight`, the transfer
from `Radiation.integrate_emission!` (and `Polarization.PolSink` for polarized frames), and
the files from `Output.generate_output_file`. What brisk light adds is when each ray step
reads the flow, which is [`FrameData`](@ref) and its `interp_field` method.
"""
module Brisklight

using Printf
using ..Iharm
using ..Grid
using ..Radiation
using ..Polarization
using ..Coordinates
using ..Utils
using ..Output

export FrameData, brisk_windows, process_brisklight_images!

"""
    FrameData(snaps, shift, tA, tB)

A structure containing the data needed to interpolate the flow for a single frame of brisk light.
"""
struct FrameData{V,S,T}
    snaps::V
    shift::S
    tA::T
    tB::T
end

Base.getindex(d::FrameData, k) = d.snaps[k]


"""
    interp_field(f, zone, X, model, data)
Interpolate a field `f` at a given zone and coordinates `X` using the flow data in `data`, which is a `FrameData` object. The interpolation is done in time
between the snapshots contained in `data.snaps`, with a time shift applied from `data.shift`. The interpolation is clamped to the time range defined by `data.tA` and `data.tB`.
"""
@inline function Iharm.interp_field(f::F, zone, X, model::Iharm.IharmParams, data::FrameData) where
{F}
    s = data.snaps
    length(s) == 1 && return Grid.interp_scalar_time(zone, f(s[1]), f(s[1]), 0.0, false)
    t = clamp(X[1] + data.shift, data.tA, data.tB)
    k = 1                      # the snapshots around t are k and k + 1
    for i in 2:length(s)-1
        s[i].t <= t && (k = i)
    end
    w = clamp(1.0 - (t - s[k].t) / (s[k+1].t - s[k].t), 0.0, 1.0)
    return Grid.interp_scalar_time(zone, f(s[k]), f(s[k+1]), w, true)
end


"""
    brisk_windows(all_geodesics, nsteps, model, n_bands, p)

Compute the time windows for each image band based on the geodesics and their equatorial crossings in the emitting region. 
The windows are determined by the shortest interval containing a fraction `p` of the crossing times.
"""
function brisk_windows(all_geodesics, nsteps, model, n_bands, p)
    Rh = 1 + sqrt(1 - model.a^2)
    nx = size(nsteps, 1)
    # Filled per image column and merged afterwards, so threads never share an array.
    samples = [[Float64[] for _ in 0:n_bands] for _ in 1:nx]
    t_lo, t_hi = fill(Inf, nx), fill(-Inf, nx)
    Threads.@threads for i in 1:nx
        for j in axes(nsteps, 2)
            traj = all_geodesics[i, j]
            band = 0
            r, th = Coordinates.bl_coord(traj[1].X, model)
            z_prev = r * cos(th)
            for k in 2:nsteps[i, j]
                X = traj[k].X
                r, th = Coordinates.bl_coord(X, model)
                z = r * cos(th)
                k > 2 && Radiation.z_turn(traj, k - 1, model) && (band += 1)
                if Radiation.radiating_region(X, model, Rh)
                    t_lo[i], t_hi[i] = min(t_lo[i], X[1]), max(t_hi[i], X[1])
                    # crossed the equatorial plane
                    z * z_prev < 0 && push!(samples[i][min(band, n_bands) + 1], (X[1] + traj[k-1].X[1]) / 2)
                end
                z_prev = z
            end
        end
    end

    # Shortest interval of the sorted times ts holding a fraction q of them.
    function shortest(ts, q)
        k = clamp(ceil(Int, q * length(ts)), 1, length(ts)) - 1
        i = argmin(i -> ts[i+k] - ts[i], 1:length(ts)-k)
        return ts[i], ts[i+k]
    end

    windows = Tuple{Float64,Float64}[]
    t_ref = NaN
    for b in 0:n_bands
        ts = sort!(reduce(vcat, (s[b+1] for s in samples)))
        if isempty(ts)
            println("Brisk light: band $b has no equatorial crossing in the emitting region; skipped.")
            continue
        end
        middle = sum(shortest(ts, 0.5)) / 2
        b == 0 && (t_ref = middle)
        push!(windows, p == 0 ? (middle, middle) : p == 1 ? (minimum(t_lo), maximum(t_hi)) : shortest(ts, p))
        @printf("Brisk light: band %d, %d crossings, window [%.2f, %.2f] M\n", b, length(ts), windows[end]...)
    end
    isnan(t_ref) && error("Brisk light: the direct image (band 0) never crosses the equatorial plane in the emitting region")
    return windows, t_ref
end


"""
    process_brisklight_images!(all_geodesics, nsteps, windows, t_ref, dump_files, load_dump, model,
        freq, meta; cadence, t_start=nothing, n_frames=nothing, fast=false, pol=false, qu_conv=0, max_lens = nothing,
        output_dir=".")

Render a movie from a dump series over stored rays: one frame every `cadence` (M), each
loading only the dumps it needs with `load_dump(path)` (kept while later frames still need them).
"""
function process_brisklight_images!(all_geodesics, nsteps, windows, t_ref, dump_files, load_dump, model,
        freq, meta; cadence, t_start=nothing, n_frames=nothing, fast=false, pol=false, qu_conv=0, max_lens = nothing, output_dir=".")
    times = [Utils.dump_time(f) for f in dump_files]
    lo, hi = minimum(first, windows), maximum(last, windows)
    t_first, t_last = times[1] - lo + t_ref, times[end] - hi + t_ref
    k0 = searchsortedfirst(times, t_first)
    t_f = something(t_start, k0 <= length(times) && times[k0] <= t_last ? times[k0] : t_first)
    mkpath(output_dir)

    nx, ny = size(nsteps)
    Image = zeros(nx, ny)
    pol_image = pol ? zeros(Polarization.NIMG, nx, ny) : nothing
    lens = max_lens === nothing ? nothing : zeros(nx, ny, max_lens + 1)
    loaded = Dict{Int,Any}()
    nframe = 0
    while t_f <= t_last + 1e-9 && (n_frames === nothing || nframe < n_frames)
        shift = t_f - t_ref
        # the dumps around each window, shifted to this frame
        needed = sort!(unique!([k for (a, b) in windows
                                for k in clamp(searchsortedlast(times, a + shift), 1, length(times)):
                                        clamp(searchsortedfirst(times, b + shift), 1, length(times))]))
        foreach(k -> k < needed[1] && delete!(loaded, k), collect(keys(loaded)))
        snaps = map(k -> get!(() -> load_dump(dump_files[k]), loaded, k), needed)
        tA, tB = fast ? (t_f, t_f) : (times[needed[1]], times[needed[end]])
        data = FrameData(snaps, shift, tA, tB)
        sink = pol ? Polarization.PolSink(data, pol_image, qu_conv) : data

        elapsed = @elapsed Threads.@threads for i in 1:nx
            for j in 1:ny
                Radiation.integrate_emission!(all_geodesics[i, j], nsteps[i, j], Image, i, j, freq, model.a, model, sink, lens)
            end
        end
        out = copy(meta)
        out["image"] = Image .* freq^3
        out["img_time"] = t_f
        out["data"] = snaps[1]
        pol && (out["pol"] = pol_image)
        lens === nothing || (out["lens"] = lens .* freq^3)
        file_name = joinpath(output_dir, @sprintf("Image.%07.1f.h5", t_f))
        Output.generate_output_file(file_name, out; format="ipole")
        @printf("Frame t = %.1f M: %d dump(s), rendered in %.2f s -> %s\n", t_f, length(snaps), elapsed, file_name)
        t_f += cadence
        nframe += 1
    end
    return nothing
end

end