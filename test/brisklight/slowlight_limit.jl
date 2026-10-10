# ---------------------------------------------------------------------------
# Brisk light must converge to slow light as p -> 1.
#
# At p = 1 every band's retained interval is (-Inf, Inf), so the clipping map is
# the identity and what remains is exactly slow light's rigid time shift. This
# test isolates that claim: it renders the same geodesics, over the same loaded
# snapshots, through the same transfer loop, with only the time map swapped --
# so the two results must agree *bitwise*, not approximately.
#
# It then sweeps p and checks the two properties that make brisk light useful:
# the error against slow light falls monotonically with p, and the number of
# dump files per band rises monotonically -- the latter matching
# ceil(width/dT) + 1, which is the statement that p controls I/O through the
# geometry rather than through a parameter.
#
# The dump series is subsampled to a coarser cadence so that p = 1 -- which by
# construction needs the entire delay support resident, exactly as slow light
# does -- fits in memory on a workstation.
#
# Usage:  julia --project=scripts --threads=auto test/brisklight/slowlight_limit.jl [dumpdir]
# ---------------------------------------------------------------------------
using Jipole
using Printf

const BL = Jipole.Brisklight
const TI = Jipole.TimeInterp

dumpdir = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "..", "..", "..", "FluidDumps")
isdir(dumpdir) || error("dump directory not found: $dumpdir")

# Geometry chosen so the p = 1 support fits the series; see the scan in the
# plan. rmin_geo cuts the plunging region, whose near-critical rays otherwise
# set the support from a handful of extreme pixels.
const MBH      = 6.2e9
const RO       = 25.0
const RMAXG    = 12.0
const RMING    = 4.0
const THETA    = 163.0
const NX       = 40
const NY       = 40
const FOV      = 60.0
const FREQ     = 230e9
const MAXNSTEP = 25000
const NBANDS   = 1            # both bands must be resolvable at this size; see the check below
const RHIGH    = 20.0
const STRIDE   = 4            # dump subsampling: dT 1 M -> 4 M
const PS       = (0.0, 0.25, 0.5, 0.75, 1.0)

all_dumps = Jipole.Utils.resolve_dump_files(dumpdir, -Inf, Inf)
dumps = all_dumps[1:STRIDE:end]
times = BL.dump_times_of(dumps)
const DT = times[2] - times[1]
@printf("series: %d of %d dumps (stride %d), t = %.2f .. %.2f M, dT = %.3f M\n",
        length(dumps), length(all_dumps), STRIDE, times[1], times[end], DT)

reader = BL.reader_for(dumps[1])
model = reader.read_header(dumps[1], MBH; th_beg = 1.74e-2, Rlow = 1.0, Rhigh = RHIGH,
                           beta_crit = 1.0, sigma_cut = 1.0, sigma_cut_high = -1.0,
                           M_unit = Jipole.Constants.M_UNIT_SANE,
                           rmin_geo = RMING, rmax_geo = RMAXG)
@printf("emitting region: %.2f < r < %.2f M\n", model.rmin_geo, model.rmax_geo)

Rh = 1 + sqrt(1.0 - model.a^2)
SourceD = 16.9e6 * Jipole.Constants.PC
fovx = SourceD / model.L_unit / Jipole.Constants.MUAS_PER_RAD * FOV / RO

geos, nsteps, crossings, Xcam, bounds = Jipole.Imaging.trace_all_geodesics(
    model, RO, THETA, 0.0, FREQ, NX, NY, fovx, fovx, MAXNSTEP, Rh, 0.0, 0.0; progress = false)
@printf("traced %d pixels, mean %.0f steps/ray; full ray depth %.1f M\n",
        NX * NY, sum(nsteps) / length(nsteps), abs(bounds.t_last))

series = BL.DumpSeries(dumps; times = times)
ref_data = reader.load_data(dumps[div(length(dumps), 2)], RHIGH, model)
cache, sample = BL.band_walk(crossings, geos, nsteps, NX, NY, NBANDS, model, Rh;
                             sample_stride = 2, weighting = :emission,
                             ref_data = ref_data, freq = FREQ)
ref_data = nothing
GC.gc()

nmse(a, b) = sum((a .- b) .^ 2) / sum(a .^ 2)

# A band with too few samples is rendered unclipped (= slow light), which is
# accurate but costs it the full snapshot support -- so it would mask the very
# p-vs-I/O trend this test is checking. Fail loudly instead of measuring noise.
let probe = BL.band_stats(sample, 0.5; verbose = false)
    bad = findall(!, probe.valid)
    isempty(bad) || error("""
        band(s) n = $(join(bad .- 1, ", ")) are unresolved at $(NX)x$(NY) with
        rmin_geo = $RMING: $(probe.nsamples) weighted samples per band. Lower NBANDS,
        raise the resolution, or lower sample_stride before interpreting this test.""")
end

# --- the p = 1 reference ---------------------------------------------------
# Build the frame at p = 1 first: its snapshot set is the widest any p needs, so
# every lower p is a subset of it and all the renders below can share it.
stats1 = BL.band_stats(sample, 1.0; verbose = false)
lo1, hi1 = BL.retained_support(stats1)
t_first, t_last = BL.frame_window(stats1, times)
@printf("p=1 support = [%.2f, %.2f] M (width %.2f M); usable frame window = [%.2f, %.2f] M\n",
        lo1, hi1, hi1 - lo1, t_first, t_last)
t_first <= t_last || error("""
    The p = 1 support ($(round(hi1 - lo1, digits=1)) M) does not fit the series
    ($(round(times[end] - times[1], digits=1)) M). Lower RO/RMAXG or raise RMING.""")

# Put the frame on a dump time, so p = 0's n = 0 band needs a single file.
const T_FRAME = times[searchsortedfirst(times, t_first)]
@printf("frame time t_frame = %.4f M\n\n", T_FRAME)

idx1, perband1 = BL.frame_dump_indices(stats1, T_FRAME, times)
@printf("p=1 needs %d snapshot(s) (%.1f GB); loading...\n",
        length(idx1), length(idx1) * 470e6 / 2^30)
snaps_all = BL.frame_snapshots!(series, idx1, RHIGH, model)
tA, tB = times[first(idx1)], times[last(idx1)]

# Slow light: the same loop, the same snapshots, the same band cache, with the
# clipping map replaced by the identity. The band cache is kept so that both
# renders take byte-for-byte the same path through `integrate_brisk!`; the only
# difference is lo/hi, which are +/-Inf on both sides at p = 1.
I_slow = zeros(Float64, NX, NY)
BL.render_brisk_frame!(I_slow, geos, nsteps, model, NX, NY, FREQ, cache,
                       BL.slowlight_reference_timemap(stats1, T_FRAME, tA, tB),
                       snaps_all; progress = false)
I_slow .*= FREQ^3

# --- sweep p --------------------------------------------------------------
println("    p    width(n=0,1,2) M        dumps/band   snaps   NMSE vs slow    max rel diff")
results = NamedTuple[]
for p in PS
    st = BL.band_stats(sample, p; verbose = false)
    idx, perband = BL.frame_dump_indices(st, T_FRAME, times)
    widths = [(w = BL.band_interval(st, b); w[2] - w[1]) for b in 1:(NBANDS + 1)]

    # Every p's dumps are a subset of p = 1's, so reuse the resident set rather
    # than reloading: this isolates the comparison from I/O entirely.
    snaps = BL.frame_snapshots!(series, idx, RHIGH, model)
    tm = BL.build_timemap(st, T_FRAME, times[first(idx)], times[last(idx)])
    I = zeros(Float64, NX, NY)
    BL.render_brisk_frame!(I, geos, nsteps, model, NX, NY, FREQ, cache, tm, snaps;
                           progress = false)
    I .*= FREQ^3

    e = nmse(I_slow, I)
    d = maximum(abs.(I .- I_slow)) / maximum(I_slow)
    @printf("%5.2f   %-22s %-12s %5d   %12.4e   %12.4e\n", p,
            join([isfinite(w) ? @sprintf("%.1f", w) : "inf" for w in widths], ","),
            join([length(r) for r in perband], "|"), length(idx), e, d)
    push!(results, (p = p, nmse = e, maxdiff = d, nsnaps = length(idx),
                    dumps_per_band = [length(r) for r in perband], widths = widths))
end

# --- assertions -----------------------------------------------------------
println()
r1 = results[end]
@printf("[1] p = 1 vs slow light: NMSE = %.3e, max rel diff = %.3e\n", r1.nmse, r1.maxdiff)
@assert r1.maxdiff == 0.0 "p = 1 is not bitwise slow light (max rel diff $(r1.maxdiff))"
println("    exact -- at p = 1 the clipping map is the identity and brisk light IS slow light")

errs = [r.nmse for r in results]
@printf("[2] NMSE vs slow light across p: %s\n", join([@sprintf("%.3e", e) for e in errs], "  "))
@assert issorted(errs; rev = true) "NMSE is not monotonically decreasing in p: $errs"
println("    monotone -- the approximation improves as p grows, as it must")

ns = [r.nsnaps for r in results]
@printf("[3] snapshots per image across p: %s\n", join(ns, "  "))
@assert issorted(ns) "snapshot count is not monotonically increasing in p: $ns"
@assert ns[1] < ns[end] "p has no effect on the snapshot count ($ns): p is not driving I/O"
println("    monotone -- p controls how many dump files are used")

# p drives the count through the geometry: dumps = ceil(width/dT) + 1.
println("[4] dumps per band vs ceil(width/dT) + 1:")
for r in results
    r.p == 1.0 && continue   # infinite width, clamped to the series
    pred = [isfinite(w) ? min(ceil(Int, w / DT) + 1, length(times)) : length(times) for w in r.widths]
    @printf("    p = %4.2f  actual %-10s predicted %-10s %s\n", r.p,
            join(r.dumps_per_band, "|"), join(pred, "|"),
            all(abs.(r.dumps_per_band .- pred) .<= 1) ? "ok" : "MISMATCH")
    @assert all(abs.(r.dumps_per_band .- pred) .<= 1) "dumps/band $(r.dumps_per_band) does not match ceil(width/dT)+1 = $pred at p = $(r.p)"
end
println("    the snapshot count follows the retained width and the dump spacing")

@printf("\n[5] band 0 at p = 0 used %d dump(s) (frame is on a dump time)\n", results[1].dumps_per_band[1])
@assert results[1].dumps_per_band[1] == 1 "band 0 at p=0 on a dump time should use 1 dump"

println("\nALL SLOW-LIGHT LIMIT TESTS PASSED")
