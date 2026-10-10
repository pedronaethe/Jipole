# ---------------------------------------------------------------------------
# Brisk light at p = 0 must reproduce fast light exactly on n = 0 pixels.
#
# A ray that crosses the equatorial plane at most once lies entirely in band 0
# (the far-end tail is capped at `midplane_crossings - 1`), and at p = 0 band 0
# is clipped to the single time `t_frame`. If `t_frame` is a dump time, the
# interpolation weight is exactly 1 or exactly 0, so the band reads one snapshot
# with no blending -- which is precisely what fast light does. The agreement is
# therefore bitwise, not approximate, and this test asserts that.
#
# Pixels with more crossings interpolate across lensing bands and must NOT
# match; the test reports their disagreement to confirm the bands are live.
#
# Usage:  julia --project=scripts --threads=auto test/brisklight/exactness.jl [dumpdir]
# ---------------------------------------------------------------------------
using Jipole
using Printf
using StaticArrays

const BL = Jipole.Brisklight
const TI = Jipole.TimeInterp

dumpdir = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "..", "..", "..", "FluidDumps")
isdir(dumpdir) || error("dump directory not found: $dumpdir")

# --- configuration ---------------------------------------------------------
# Small and cheap: this is an exactness test, not a physics run.
const MBH      = 6.2e9
const RO       = 100.0          # reduced so the delay spread fits the series
const THETA    = 163.0
const PHI      = 0.0
const NX       = 48
const NY       = 48
const FOV      = 80.0
const FREQ     = 230e9
const MAXNSTEP = 25000
const NBANDS   = 2
const RHIGH    = 20.0
const RLOW     = 1.0
const SRC_PC   = 16.9e6

dumps = Jipole.Utils.resolve_dump_files(dumpdir, -Inf, Inf)
times = BL.dump_times_of(dumps)
@printf("%d dumps, t = %.3f .. %.3f M, dT = %.4f M\n",
        length(dumps), times[1], times[end], times[2] - times[1])

# Anchor the frame on a dump in the middle of the series, so bands n>0 (which
# look back in time) still have snapshots available.
const KREF    = clamp(round(Int, 0.75 * length(dumps)), 1, length(dumps))
const T_FRAME = times[KREF]
@printf("reference dump: index %d (file %s), t_frame = %.6f M\n",
        KREF, basename(dumps[KREF]), T_FRAME)

reader = BL.reader_for(dumps[KREF])
model = reader.read_header(dumps[KREF], MBH; th_beg = 1.74e-2, Rlow = RLOW, Rhigh = RHIGH,
                           beta_crit = 1.0, sigma_cut = 1.0, sigma_cut_high = -1.0,
                           M_unit = Jipole.Constants.M_UNIT_SANE)

Rh      = 1 + sqrt(1.0 - model.a^2)
SourceD = SRC_PC * Jipole.Constants.PC
DX      = SourceD / model.L_unit / Jipole.Constants.MUAS_PER_RAD * FOV
fovx    = DX / RO
fovy    = DX / RO

# --- trace once, share between both renders --------------------------------
geos, nsteps, crossings, Xcam, bounds = Jipole.Imaging.trace_all_geodesics(
    model, RO, THETA, PHI, FREQ, NX, NY, fovx, fovy, MAXNSTEP, Rh, 0.0, 0.0;
    progress = false)
@printf("traced %d pixels; tgeof = %.3f  tgeoi = %.3f  t_last = %.3f M\n",
        NX * NY, bounds.tgeof, bounds.tgeoi, bounds.t_last)
println("crossing histogram: ", [(c, count(==(c), crossings)) for c in sort(unique(crossings))])

# --- fast light on the single reference dump -------------------------------
snap_ref = reader.load_data(dumps[KREF], RHIGH, model)
I_fast = zeros(Float64, NX, NY)
let bad = Threads.Atomic{Int}(0)
    # Drive the shared transfer loop with a 1-snapshot set and a constant time
    # map: that IS fast light, and routing it through the same code proves the
    # unified path has not changed fast-light behaviour.
    ss1 = TI.SnapshotSet([snap_ref])
    tm1 = TI.fastlight_timemap(T_FRAME, T_FRAME, T_FRAME)
    BL.render_brisk_frame!(I_fast, geos, nsteps, model, NX, NY, FREQ,
                           nothing, tm1, ss1; progress = false)
    I_fast .*= FREQ^3
end

# Cross-check that against the untouched upstream fast-light entry point.
I_ref = Jipole.Imaging.raytrace_image(model, [snap_ref], RO, THETA, PHI, FREQ,
                                      NX, NY, fovx, fovy, MAXNSTEP, Rh, 0.0, 0.0)
dref = maximum(abs.(I_ref .- I_fast)) / maximum(I_ref)
@printf("\n[0] unified loop vs Imaging.raytrace_image : max rel diff = %.3e\n", dref)
@assert dref == 0.0 "the unified transfer loop changed fast-light results (rel diff $dref)"
println("    exact -- the shared loop reproduces upstream fast light bitwise")

# --- brisk light at p = 0 --------------------------------------------------
stats, cache, sample = BL.analyze_bands(
    crossings, geos, nsteps, NX, NY, NBANDS, model, 0.0;
    sample_stride = 1, ref_data = snap_ref, freq = FREQ, verbose = true)

series = BL.DumpSeries(dumps; times = times)
idx, per_band = BL.frame_dump_indices(stats, T_FRAME, times)
@printf("\nframe needs %d snapshot(s): dumps %s\n", length(idx), string(idx))
println("dumps per band (n=0,1,2): ", [length(r) for r in per_band])
@assert length(per_band[1]) == 1 "band 0 should need exactly 1 dump at p=0 on a dump time, got $(length(per_band[1]))"

snaps = BL.frame_snapshots!(series, idx, RHIGH, model)
tm = BL.build_timemap(stats, T_FRAME, times[first(idx)], times[last(idx)])
I_brisk = zeros(Float64, NX, NY)
BL.render_brisk_frame!(I_brisk, geos, nsteps, model, NX, NY, FREQ,
                       cache, tm, snaps; progress = false)
I_brisk .*= FREQ^3

# --- compare ---------------------------------------------------------------
peak = maximum(I_fast)
mask0 = crossings .<= 1              # entirely band 0
maskN = crossings .>= 2              # interpolate across bands
nmse(a, b) = sum((a .- b) .^ 2) / sum(a .^ 2)

d0 = maximum(abs.(I_brisk[mask0] .- I_fast[mask0])) / peak
@printf("\n[1] n=0 pixels (%d of %d): max rel diff = %.3e\n", count(mask0), NX * NY, d0)
@assert d0 == 0.0 "p=0 is not bitwise fast light on n=0 pixels (rel diff $d0)"
println("    exact -- p = 0 reproduces fast light on n = 0 pixels bitwise")

if count(maskN) > 0
    dN = maximum(abs.(I_brisk[maskN] .- I_fast[maskN])) / peak
    @printf("[2] n>=1 pixels (%d): max rel diff = %.3e, NMSE = %.3e\n",
            count(maskN), dN, nmse(I_fast[maskN], I_brisk[maskN]))
    @assert dN > 0.0 "n>=1 pixels are identical to fast light: the lensing bands are not doing anything"
    println("    nonzero -- higher bands are evaluated at their own times, as intended")
end

@printf("[3] whole image NMSE vs fast light = %.4e\n", nmse(I_fast, I_brisk))

# --- off-grid frame: the fast-light cadence path ---------------------------
# Half a dump spacing off the grid. Band 0 now has to interpolate, so it should
# need 2 dumps and agree with a fast-light render interpolated to the same time.
const T_OFF = T_FRAME + 0.5 * (times[2] - times[1])
idx2, per_band2 = BL.frame_dump_indices(stats, T_OFF, times)
@printf("\n[4] off-grid frame t_frame = %.6f M: band 0 needs %d dump(s)\n",
        T_OFF, length(per_band2[1]))
@assert length(per_band2[1]) == 2 "band 0 off-grid should need 2 dumps, got $(length(per_band2[1]))"

snaps2 = BL.frame_snapshots!(series, idx2, RHIGH, model)
tm2 = BL.build_timemap(stats, T_OFF, times[first(idx2)], times[last(idx2)])
I_brisk2 = zeros(Float64, NX, NY)
BL.render_brisk_frame!(I_brisk2, geos, nsteps, model, NX, NY, FREQ,
                       cache, tm2, snaps2; progress = false)
I_brisk2 .*= FREQ^3

# Fast light interpolated to T_OFF: constant time map over the bracketing pair.
kk = BL.band_dump_range(T_OFF, T_OFF, times)
ssf = BL.frame_snapshots!(series, collect(kk), RHIGH, model)
I_fast2 = zeros(Float64, NX, NY)
BL.render_brisk_frame!(I_fast2, geos, nsteps, model, NX, NY, FREQ, nothing,
                       TI.fastlight_timemap(T_OFF, times[first(kk)], times[last(kk)]),
                       ssf; progress = false)
I_fast2 .*= FREQ^3

d0b = maximum(abs.(I_brisk2[mask0] .- I_fast2[mask0])) / maximum(I_fast2)
@printf("    n=0 pixels vs interpolated fast light: max rel diff = %.3e\n", d0b)
@assert d0b == 0.0 "off-grid p=0 does not match interpolated fast light on n=0 pixels ($d0b)"
println("    exact -- synchronization holds off the dump grid too")

println("\nALL EXACTNESS TESTS PASSED")
