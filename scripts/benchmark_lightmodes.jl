# ---------------------------------------------------------------------------
# Compare fast, brisk and slow light on one geometry and write a metrics table.
#
# The question this answers is whether brisk light actually buys anything, so it
# measures the two things that could pay for the approximation -- I/O volume and
# resident memory -- alongside the error it introduces, and reports them against
# a slow-light reference rendered through the *same* transfer loop. Comparing
# runtime alone would be misleading: brisk light does roughly the same number of
# emissivity evaluations as slow light per frame, and its saving is in how many
# snapshots have to be read and held.
#
# All three modes are put on the same observer-time grid, so rows are directly
# comparable frame by frame.
#
#   julia --project=scripts --threads=auto scripts/benchmark_lightmodes.jl bench.toml
#
# Config: everything main.jl reads, plus
#
#   [benchmark]
#   p_values = [0.0, 0.25, 0.5, 0.75, 1.0]
#   dump_stride = 1      # use every Nth dump (coarsens dT, cuts memory)
#   n_frames = 3
#   include_fast = true
#   include_slow = true  # the shared-loop slow-light reference
#   csv = "benchmark.csv"
#   save_images = false
# ---------------------------------------------------------------------------
using Jipole
using Printf
using TOML
using Statistics: mean

const BL = Jipole.Brisklight
const TI = Jipole.TimeInterp

length(ARGS) == 1 || error("usage: julia --project=scripts benchmark_lightmodes.jl config.toml")
config = TOML.parsefile(ARGS[1])
cfg(sec, key, def) = haskey(config, sec) ? get(config[sec], key, def) : def

const MBH      = cfg("physical", "MBH", 6.2e9)
const m_unit   = cfg("physical", "m_unit", "MAD")
const M_unit   = m_unit == "MAD" ? Jipole.Constants.M_UNIT_MAD :
                 m_unit == "SANE" ? Jipole.Constants.M_UNIT_SANE : float(m_unit)
const dumpdir  = expanduser(cfg("dump", "dump_filepath", ""))
const Rhigh    = cfg("plasma", "Rhigh", 20.0)
const Rlow     = cfg("plasma", "Rlow", 1.0)
const beta_crit = cfg("plasma", "beta_crit", 1.0)
const th_beg   = cfg("plasma", "th_beg", 1.74e-2)
const sigma_cut = cfg("plasma", "sigma_cut", 1.0)
const sigma_cut_high = cfg("plasma", "sigma_cut_high", -1.0)
const rmin_geo = cfg("plasma", "rmin_geo", nothing)
const rmax_geo = cfg("plasma", "rmax_geo", nothing)
const ro       = cfg("camera", "ro", 1000.0)
const th       = cfg("camera", "theta_o", 163.0)
const phi      = cfg("camera", "phi", 0.0)
const nx       = cfg("image", "pixels_x", 64)
const ny       = cfg("image", "pixels_y", 64)
const fov_size = cfg("image", "fov_size", 160.0)
const xoff     = cfg("image", "xoff", 0.0)
const yoff     = cfg("image", "yoff", 0.0)
const freq     = cfg("observing", "freq", 230e9)
const src_pc   = cfg("observing", "source_distance_pc", 16.9e6)
const maxnstep = cfg("raytracing", "maxnstep", 25000)
const n_bands  = cfg("brisklight", "n_bands", 2)
const cadence  = cfg("brisklight", "cadence", 10.0)
const stride   = cfg("brisklight", "sample_stride", 4)
const weighting = Symbol(cfg("brisklight", "weighting", "emission"))

const p_values   = Float64.(cfg("benchmark", "p_values", [0.0, 0.25, 0.5, 0.75, 1.0]))
const dump_stride = cfg("benchmark", "dump_stride", 1)
const n_frames   = cfg("benchmark", "n_frames", 3)
const incl_fast  = cfg("benchmark", "include_fast", true)
const incl_slow  = cfg("benchmark", "include_slow", true)
const csv_path   = cfg("benchmark", "csv", "benchmark.csv")
const save_images = cfg("benchmark", "save_images", false)

isdir(dumpdir) || error("[dump].dump_filepath must be a directory of dumps (got '$dumpdir')")

# --- setup -----------------------------------------------------------------
all_dumps = Jipole.Utils.resolve_dump_files(dumpdir, -Inf, Inf)
dumps = all_dumps[1:dump_stride:end]
times = BL.dump_times_of(dumps)
dT = length(times) > 1 ? times[2] - times[1] : NaN
@printf("Series: %d of %d dumps (stride %d), t = %.2f .. %.2f M, dT = %.3f M\n",
        length(dumps), length(all_dumps), dump_stride, times[1], times[end], dT)

reader = BL.reader_for(dumps[1])
model = reader.read_header(dumps[1], MBH; th_beg = th_beg, Rlow = Rlow, Rhigh = Rhigh,
    beta_crit = beta_crit, sigma_cut = sigma_cut, sigma_cut_high = sigma_cut_high,
    M_unit = M_unit, rmin_geo = rmin_geo, rmax_geo = rmax_geo)

Rh = 1 + sqrt(1.0 - model.a^2)
SourceD = src_pc * Jipole.Constants.PC
DXsize = SourceD / model.L_unit / Jipole.Constants.MUAS_PER_RAD * fov_size
fovx = DXsize / ro

geo_stat = @timed Jipole.Imaging.trace_all_geodesics(
    model, ro, th, phi, freq, nx, ny, fovx, fovx, maxnstep, Rh, xoff, yoff; progress = false)
geos, nsteps, crossings, Xcam, bounds = geo_stat.value
const TOTAL_STEPS = sum(nsteps)
@printf("Traced %d pixels in %.2f s (%.2f GB alloc), mean %.0f steps/ray, %d steps total\n",
        nx * ny, geo_stat.time, geo_stat.bytes / 2^30, TOTAL_STEPS / (nx * ny), TOTAL_STEPS)

# Bytes a snapshot occupies, and bytes the transfer touches per step. Each
# emissivity evaluation quadrilinearly interpolates 11 fields (ne, b, theta_e,
# sigma, dtheta_e/dRhigh, U1-3, B1-3) over 8 grid corners, for each of the two
# bracketing snapshots. It is an upper bound -- the cache serves most of it --
# but it is the figure a bandwidth-bound kernel would have to move.
const BYTES_PER_STEP = 11 * 8 * sizeof(Float64) * 2

ref_data = reader.load_data(dumps[div(length(dumps) + 1, 2)], Rhigh, model)
const SNAP_BYTES = BL.snapshot_bytes(ref_data)
@printf("Snapshot footprint: %.0f MB each\n", SNAP_BYTES / 2^20)

walk_stat = @timed BL.band_walk(crossings, geos, nsteps, nx, ny, n_bands, model, Rh;
    sample_stride = stride, weighting = weighting, ref_data = ref_data, freq = freq)
cache, sample = walk_stat.value
ref_data = nothing
GC.gc()
@printf("Band walk: %.2f s (%.2f GB alloc)\n", walk_stat.time, walk_stat.bytes / 2^30)

# --- the frame grid, shared by every mode ---------------------------------
# Keyed off p = 1, whose support is the widest, so one grid serves all modes and
# frames can be compared row by row.
stats_ref = BL.band_stats(sample, 1.0; verbose = false)
t_first, t_last = BL.frame_window(stats_ref, times)
t_first <= t_last || error("""
    The p = 1 support does not fit this series, so there is no grid every mode can
    share. Lower [camera].ro or [plasma].rmax_geo, raise rmin_geo, or supply more dumps.""")
# Start on a dump time when one lies inside the usable window, so that p = 0's
# n = 0 band reads a single file and lines up with fast light; otherwise start at
# the window's edge and accept interpolation there.
const schedule = let k0 = searchsortedfirst(times, t_first)
    t0 = (k0 <= length(times) && times[k0] <= t_last) ? times[k0] : t_first
    sched = Float64[]
    tt = t0
    while tt <= t_last + 1e-9 && length(sched) < n_frames
        push!(sched, tt)
        tt += cadence
    end
    sched
end
isempty(schedule) && error("""
    No frame fits in the usable window [$(round(t_first, digits=3)), $(round(t_last, digits=3))] M.
    The p = 1 support is nearly as wide as the dump series. Lower [camera].ro or
    [plasma].rmax_geo, raise [plasma].rmin_geo, or supply more dumps.""")
@printf("Frame grid: %s (%d frame(s))\n\n", join(round.(schedule, digits = 2), ", "), length(schedule))

nmse(a, b) = sum((a .- b) .^ 2) / sum(a .^ 2)

# Where the method is supposed to matter. At p = 0 band 0 is evaluated at the
# frame time, i.e. at exactly fast light's snapshot, so brisk and fast light
# agree on the direct image *by construction*. The direct image carries most of
# the flux, so a whole-image NMSE is dominated by pixels the approximation
# cannot change and badly understates what brisk light buys. The indirect pixels
# -- those whose rays cross the equatorial plane more than once, i.e. the photon
# ring -- are the ones brisk light exists to get right, so the error is reported
# on both sets.
const RING = crossings .>= 2
const NRING = count(RING)
masked_nmse(a, b) = NRING == 0 ? NaN : sum((a[RING] .- b[RING]) .^ 2) / sum(a[RING] .^ 2)
@printf("Pixels: %d total, %d indirect (n >= 1, %.1f%%)\n", nx * ny, NRING, 100 * NRING / (nx * ny))
const rows = NamedTuple[]
const images = Dict{String,Vector{Matrix{Float64}}}()

"""
Render `schedule` under one mode and return its metrics row plus its frames.
`timemap_for(stats, t_frame, tA, tB)` and `dumps_for(stats, t_frame)` are what
distinguish the modes; everything else is held fixed so the comparison is of
the time map alone.
"""
function run_mode(label, stats, dumps_for, timemap_for, band_cache)
    series = BL.DumpSeries(dumps; times = times)
    frames = Matrix{Float64}[]
    render_t = 0.0; render_bytes = 0; render_gc = 0.0
    nsnap = Int[]; resident = Int[]
    GC.gc()
    for t_frame in schedule
        idx = dumps_for(stats, t_frame)
        snaps = BL.frame_snapshots!(series, idx, Rhigh, model)
        tm = timemap_for(stats, t_frame, times[first(idx)], times[last(idx)])
        Image = zeros(Float64, nx, ny)
        st = @timed BL.render_brisk_frame!(Image, geos, nsteps, model, nx, ny, freq,
                                           band_cache, tm, snaps; progress = false)
        render_t += st.time; render_bytes += st.bytes; render_gc += st.gctime
        Image .*= freq^3
        push!(frames, Image)
        push!(nsnap, length(idx))
        push!(resident, sum(BL.snapshot_bytes(v) for v in values(series.cache); init = 0))
    end
    nf = length(frames)
    row = (label = label,
           frames = nf,
           snaps_per_image = mean(nsnap),
           max_resident_gb = maximum(resident) / 2^30,
           dump_loads = series.n_loads,
           gb_read = series.bytes_read / 2^30,
           io_seconds = series.load_time,
           render_seconds = render_t,
           render_per_frame = render_t / nf,
           render_alloc_gb = render_bytes / 2^30,
           gc_seconds = render_gc,
           steps_per_second = TOTAL_STEPS * nf / render_t,
           eff_bandwidth_gbs = TOTAL_STEPS * nf * BYTES_PER_STEP / render_t / 2^30,
           peak_rss_gb = Sys.maxrss() / 2^30)
    images[label] = frames
    GC.gc()
    return row
end

# --- fast light ------------------------------------------------------------
if incl_fast
    push!(rows, run_mode("fast", BL.band_stats(sample, 0.0; verbose = false),
        (s, tf) -> collect(BL.band_dump_range(tf, tf, times)),
        (s, tf, tA, tB) -> TI.fastlight_timemap(tf, tA, tB),
        nothing))
end

# --- brisk light, one row per p -------------------------------------------
# The band walk and the geodesics are reused across p; only the statistics are
# rebuilt, which is the whole point of separating `band_walk` from `band_stats`.
for p in p_values
    st = BL.band_stats(sample, p; verbose = false)
    widths = [(w = BL.band_interval(st, b); w[2] - w[1]) for b in 1:(n_bands + 1)]
    row = run_mode(@sprintf("brisk p=%.2f", p), st,
        (s, tf) -> first(BL.frame_dump_indices(s, tf, times)),
        BL.build_timemap, cache)
    # Fraction of steps evaluated against a frozen plasma. Above ~1% the run is
    # not a valid brisk-light run and its NMSE should not be quoted.
    clamp_frac = mean(begin
        idx = first(BL.frame_dump_indices(st, tf, times))
        BL.clamped_fraction(st, sample, tf, times[first(idx)], times[last(idx)])[1]
    end for tf in schedule)
    push!(rows, (; row..., p = p, widths = widths, clamped = clamp_frac))
end

# --- slow light, through the same loop ------------------------------------
# Same transfer loop, same band cache, same snapshots -- only the time map
# differs, so this is the reference brisk light at p = 1 must reproduce exactly.
if incl_slow
    push!(rows, run_mode("slow", stats_ref,
        (s, tf) -> first(BL.frame_dump_indices(s, tf, times)),
        BL.slowlight_reference_timemap, cache))
end

# --- report ---------------------------------------------------------------
const refkey = incl_slow ? "slow" : nothing
err_of(label) = refkey === nothing ? NaN :
    mean(nmse(images[refkey][k], images[label][k]) for k in 1:length(schedule))
ring_err_of(label) = refkey === nothing ? NaN :
    mean(masked_nmse(images[refkey][k], images[label][k]) for k in 1:length(schedule))

println("="^128)
@printf("%-14s %7s %9s %8s %8s %8s %9s %10s %8s %11s %11s\n",
        "mode", "snaps", "resid GB", "loads", "GB read", "I/O s", "render s",
        "Msteps/s", "GB/s", "NMSE all", "NMSE ring")
println("-"^128)
for r in rows
    @printf("%-14s %7.2f %9.2f %8d %8.2f %8.2f %9.2f %10.1f %8.2f %11.3e %11.3e\n",
            r.label, r.snaps_per_image, r.max_resident_gb, r.dump_loads, r.gb_read,
            r.io_seconds, r.render_seconds, r.steps_per_second / 1e6,
            r.eff_bandwidth_gbs, err_of(r.label), ring_err_of(r.label))
end
println("="^128)

if refkey !== nothing
    slow = rows[findfirst(r -> r.label == "slow", rows)]
    println("\nRelative to slow light (the reference), on the same frame grid:")
    @printf("%-14s %9s %10s %8s %9s %12s %12s\n",
            "mode", "snaps x", "GB read x", "I/O x", "render x", "NMSE all", "NMSE ring")
    for r in rows
        r.label == "slow" && continue
        @printf("%-14s %8.2fx %9.2fx %7.2fx %8.2fx %12.3e %12.3e\n", r.label,
                slow.snaps_per_image / r.snaps_per_image,
                r.gb_read > 0 ? slow.gb_read / r.gb_read : NaN,
                r.io_seconds > 0 ? slow.io_seconds / r.io_seconds : NaN,
                r.render_seconds > 0 ? slow.render_seconds / r.render_seconds : NaN,
                err_of(r.label), ring_err_of(r.label))
    end

    # The comparison that decides whether the method is worth it. Fast light is
    # free but cannot move the indirect pixels at all, so if brisk light's ring
    # error is not clearly below fast light's, the extra snapshots bought
    # nothing at this geometry -- which usually means the photon ring carries
    # too little flux here to demonstrate anything, not that the method fails.
    if incl_fast
        fastrow = rows[findfirst(r -> r.label == "fast", rows)]
        fe = ring_err_of("fast")
        println()
        @printf("Fast light's ring NMSE is %.3e using %.2f snapshot(s).\n",
                fe, fastrow.snaps_per_image)
        for r in rows
            (r.label == "slow" || r.label == "fast") && continue
            re = ring_err_of(r.label)
            @printf("  %-14s ring NMSE %.3e (%.2fx better than fast) for %.1fx the snapshots\n",
                    r.label, re, fe / re, r.snaps_per_image / fastrow.snaps_per_image)
        end
        # Error reduction per extra snapshot, over the rows that actually
        # approximate something. A row that reproduces the reference exactly has
        # zero error and would win any ratio trivially, and fast light is the baseline
        # the ratio is taken against, so both are excluded.
        cand = [r for r in rows if startswith(r.label, "brisk") && ring_err_of(r.label) > 0]
        if length(cand) > 1
            eff = [(fe / ring_err_of(r.label)) / r.snaps_per_image for r in cand]
            @printf("Best ring-error reduction per snapshot: %s (%.2fx per snapshot)\n",
                    cand[argmax(eff)].label, maximum(eff))
        end
        brisk_rows = [r for r in rows if startswith(r.label, "brisk")]
        if !isempty(brisk_rows) &&
           maximum(fe / ring_err_of(r.label) for r in brisk_rows) < 1.5
            @info """
                Brisk light is not clearly beating fast light on the ring at this geometry.
                That is expected when the indirect pixels carry little flux: enlarge
                rmax_geo, raise the resolution, or widen the FOV so the photon ring is
                properly resolved before drawing conclusions about the method."""
        end
    end
end

# --- CSV ------------------------------------------------------------------
mkpath(dirname(csv_path) == "" ? "." : dirname(csv_path))
open(csv_path, "w") do io
    println(io, "label,p,frames,pixels_x,pixels_y,total_steps,dump_stride,dT,",
                "snaps_per_image,max_resident_gb,dump_loads,gb_read,io_seconds,",
                "render_seconds,render_per_frame,render_alloc_gb,gc_seconds,",
                "steps_per_second,eff_bandwidth_gbs,peak_rss_gb,nmse_vs_slow,nmse_ring_vs_slow,clamped_fraction,",
                "threads,geodesic_seconds,bandwalk_seconds,snapshot_bytes,hdi_widths")
    for r in rows
        @printf(io, "%s,%s,%d,%d,%d,%d,%d,%.6f,%.4f,%.4f,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.1f,%.4f,%.4f,%.6e,%.6e,%s,%d,%.4f,%.4f,%d,%s\n",
                r.label, haskey(r, :p) ? string(r.p) : "", r.frames, nx, ny, TOTAL_STEPS,
                dump_stride, dT, r.snaps_per_image, r.max_resident_gb, r.dump_loads,
                r.gb_read, r.io_seconds, r.render_seconds, r.render_per_frame,
                r.render_alloc_gb, r.gc_seconds, r.steps_per_second, r.eff_bandwidth_gbs,
                r.peak_rss_gb, err_of(r.label), ring_err_of(r.label),
                haskey(r, :clamped) ? @sprintf("%.6f", r.clamped) : "",
                Threads.nthreads(), geo_stat.time, walk_stat.time, SNAP_BYTES,
                haskey(r, :widths) ? join([@sprintf("%.3f", w) for w in r.widths], "|") : "")
    end
end
println("\nWrote $csv_path")

if save_images
    outdir = joinpath(dirname(csv_path) == "" ? "." : dirname(csv_path), "frames")
    mkpath(outdir)
    for (label, frames) in images, (k, img) in enumerate(frames)
        open(joinpath(outdir, replace(label, " " => "_", "=" => "") * @sprintf(".%03d.txt", k)), "w") do io
            for i in axes(img, 1); println(io, join(view(img, i, :), '\t')); end
        end
    end
    println("Wrote frames to $outdir")
end

@printf("\nPeak RSS %.2f GB, %d thread(s), geodesics %.2f s, band walk %.2f s\n",
        Sys.maxrss() / 2^30, Threads.nthreads(), geo_stat.time, walk_stat.time)
