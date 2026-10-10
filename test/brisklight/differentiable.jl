# ---------------------------------------------------------------------------
# Brisk light must stay differentiable and generic.
#
# Jipole carries ForwardDiff Duals through the radiative transfer by keeping
# IharmParams and IharmData parametric. Brisk light adds a time map and a
# snapshot container in front of that, so this checks the derivative still comes
# out the other side, and matches a central difference.
#
# The time map is built from clamps, which are differentiable almost everywhere.
# A step clipped to the edge of its band's interval has zero derivative with
# respect to the time -- correctly so, since the plasma it reads genuinely does
# not move when that step moves. The derivative taken here is with respect to
# Rhigh, a plasma parameter, which every step's emissivity depends on.
#
# Usage:  julia --project=scripts --threads=auto test/brisklight/differentiable.jl [dumpdir]
# ---------------------------------------------------------------------------
using Jipole
using Printf
# ForwardDiff is a Jipole dependency, not a scripts/ one, so reach it through the
# module rather than adding a dep just for this test.
const ForwardDiff = Jipole.Iharm.ForwardDiff
using StaticArrays

const BL = Jipole.Brisklight
const TI = Jipole.TimeInterp

dumpdir = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "..", "..", "..", "FluidDumps")
isdir(dumpdir) || error("dump directory not found: $dumpdir")

const MBH   = 6.2e9
const RO    = 20.0
const THETA = 163.0
const NX, NY = 16, 16
const FOV   = 60.0
const FREQ  = 230e9
const NBANDS = 1
const RHIGH0 = 20.0
const STRIDE = 8

all_dumps = Jipole.Utils.resolve_dump_files(dumpdir, -Inf, Inf)
dumps = all_dumps[1:STRIDE:end]
times = BL.dump_times_of(dumps)
reader = BL.reader_for(dumps[1])

base = reader.read_header(dumps[1], MBH; th_beg = 1.74e-2, Rlow = 1.0, Rhigh = RHIGH0,
    beta_crit = 1.0, sigma_cut = 1.0, sigma_cut_high = -1.0,
    M_unit = Jipole.Constants.M_UNIT_SANE, rmin_geo = 4.0, rmax_geo = 10.0)
Rh = 1 + sqrt(1.0 - base.a^2)
SourceD = 16.9e6 * Jipole.Constants.PC
fovx = SourceD / base.L_unit / Jipole.Constants.MUAS_PER_RAD * FOV / RO

geos, nsteps, crossings, Xcam, bounds = Jipole.Imaging.trace_all_geodesics(
    base, RO, THETA, 0.0, FREQ, NX, NY, fovx, fovx, 25000, Rh, 0.0, 0.0; progress = false)

# Load the frame's snapshots once in Float64 and keep a context for each: the
# raw primitives never depend on Rhigh, so only the derived fields have to be
# rebuilt per evaluation. This is also how master's gradient path works -- a Dual
# cannot enter through read_header, whose builder has concrete Float64 fields.
loaded = [reader.load_data(dumps[1], RHIGH0, base)]
cache, sample = BL.band_walk(crossings, geos, nsteps, NX, NY, NBANDS, base, Rh;
    sample_stride = 1, weighting = :emission, ref_data = loaded[1], freq = FREQ)

stats = BL.band_stats(sample, 0.5; verbose = false)
t_first, t_last = BL.frame_window(stats, times)
t_first <= t_last || error("no frame fits; shrink the geometry or add dumps")
const T_FRAME = times[max(1, searchsortedfirst(times, t_first))]
const IDX = first(BL.frame_dump_indices(stats, T_FRAME, times))
@printf("frame t_frame = %.3f M, %d snapshot(s), %d pixels\n", T_FRAME, length(IDX), NX * NY)

const CTXS = [Jipole.Iharm.grmhd_context(base, reader.load_data(dumps[k], RHIGH0, base))
              for k in IDX]
loaded = nothing
GC.gc()

const MU = Jipole.Constants.M_UNIT_SANE

"""
Total brisk-light flux as a function of `Rhigh`.

`Rhigh` sets the electron temperature in the strongly magnetized region, so it
enters every snapshot's derived fields and every step's emissivity. Rebuilding
the model and all the frame's snapshots at the argument's type exercises the
whole generic chain: `SnapshotSet`, `TimeMap`, `nearest_pair` and the transfer
loop all have to carry the Dual through.
"""
function total_flux(Rhigh)
    T = promote_type(typeof(Rhigh), Float64)
    pairs = [Jipole.Iharm.dual_model_data(ctx, T, MBH, MU, 1.0, Rhigh,
                                          1.0, 1.74e-2, 1.0, -1.0) for ctx in CTXS]
    model_d = first(pairs[1])
    snaps = TI.SnapshotSet([last(pr) for pr in pairs], [times[k] for k in IDX])
    tm = BL.build_timemap(stats, T_FRAME, times[first(IDX)], times[last(IDX)])
    Image = zeros(T, NX, NY)
    BL.render_brisk_frame!(Image, geos, nsteps, model_d, NX, NY, FREQ, cache, tm, snaps;
                           progress = false)
    return sum(Image) * FREQ^3
end

f0 = total_flux(RHIGH0)
@printf("\nflux(Rhigh = %.1f) = %.8e\n", RHIGH0, f0)
@assert isfinite(f0) && f0 > 0 "brisk light produced no flux; nothing to differentiate"

ad = ForwardDiff.derivative(total_flux, RHIGH0)
@printf("d(flux)/d(Rhigh)  ForwardDiff = %+.8e\n", ad)
@assert isfinite(ad) "the ForwardDiff derivative is not finite"
@assert ad != 0 "the derivative is exactly zero; the Dual is not reaching the transfer loop"

# Central difference. The step is a compromise: the transfer is smooth in Rhigh,
# but the sum over pixels loses precision, so h too small is swamped by roundoff.
h = 1e-4 * RHIGH0
fd = (total_flux(RHIGH0 + h) - total_flux(RHIGH0 - h)) / (2h)
rel = abs(ad - fd) / abs(fd)
@printf("d(flux)/d(Rhigh)  central diff = %+.8e  (h = %.2e)\n", fd, h)
@printf("relative difference = %.3e\n", rel)
@assert rel < 1e-5 "AD and finite differences disagree by $rel; the derivative is wrong"
println("\n[1] ForwardDiff matches central differences through the brisk-light render")

# Genericity: the derived fields and the containers must follow the input type
# rather than being pinned to Float64 anywhere in the chain.
let d = ForwardDiff.Dual{Nothing,Float64,1}(RHIGH0, ForwardDiff.Partials((1.0,)))
    T = typeof(d)
    model_d, data_d = Jipole.Iharm.dual_model_data(CTXS[1], T, MBH, MU, 1.0, d,
                                                   1.0, 1.74e-2, 1.0, -1.0)
    snaps = TI.SnapshotSet([data_d], [times[IDX[1]]])
    @printf("[2] derived field eltype with a Dual Rhigh: %s\n", eltype(data_d.θe))
    @assert eltype(data_d.θe) <: ForwardDiff.Dual "derived fields did not promote to Dual"
    @printf("    raw primitive eltype (shared, must stay Float64): %s\n", eltype(data_d.RHO))
    @assert eltype(data_d.RHO) == Float64 "raw primitives were needlessly promoted"
    a, b, w = TI.nearest_pair(snaps, times[IDX[1]])
    @printf("    nearest_pair weight type: %s\n", typeof(w))
    @printf("    model Rhigh type: %s\n", typeof(model_d.Rhigh))
    @assert typeof(model_d.Rhigh) <: ForwardDiff.Dual
end
println("    the snapshot container and time map stay generic in the element type")

# The time map itself must be differentiable in the frame time, and the clamp
# must behave: a step inside its band's interval moves with the frame, one
# clipped to the edge does not.
let tm = BL.build_timemap(stats, T_FRAME, times[first(IDX)], times[last(IDX)])
    lo, hi = BL.band_interval(stats, 1)
    inside = 0.5 * (lo + hi)
    outside = lo - 10.0
    din = ForwardDiff.derivative(tf -> TI.map_time(
        BL.build_timemap(stats, tf, times[first(IDX)], times[last(IDX)]), inside, 0), T_FRAME)
    dout = ForwardDiff.derivative(tf -> TI.map_time(
        BL.build_timemap(stats, tf, times[first(IDX)], times[last(IDX)]), outside, 0), T_FRAME)
    @printf("[3] d(t_abs)/d(t_frame): step inside the interval = %.1f, clipped step = %.1f\n",
            din, dout)
    @assert din == 1.0 "an unclipped step should track the frame time exactly"
    @assert dout == 1.0 "a clipped step should still track the frame, which shifts the interval with it"
end
println("    map_time is differentiable in the frame time")

println("\nALL DIFFERENTIABILITY TESTS PASSED")
