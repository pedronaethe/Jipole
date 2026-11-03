# Slow light must never extrapolate the flow in time: Iharm.set_tinterp_ns returns a weight in [0, 1] for any time,
# inside or outside the three-snapshot window, as ipole's set_tinterp_ns does. Both slow-light methods: the CPU
# `Vector` window (model.slow_light) and the GPU `NTuple{3}` window.
# Regression for the M87 crash (Oct 19): the clamp had been left unreachable by a merge, a step ending 3 M past the
# last snapshot was extrapolated, and theta_e came out negative. See crash_replay.jl for the full replay.
# Usage: julia --project=scripts tinterp_clamp.jl DUMPDIR   (any dump of the directory, for the grid header)
using Jipole, Printf

const IH = Jipole.Iharm
dumpdir = ARGS[1]
dump = first(sort(filter(f -> endswith(f, ".phdf") || endswith(f, ".h5"), readdir(dumpdir; join=true))))
reader = Jipole.Kharma.is_kharma_dump(dump) ? Jipole.Kharma : IH
model = reader.read_header(dump, 4.14e6; slow_light=true)
X(t) = (t, 0.0, 0.0, 0.0)

# Only the snapshot times enter the weight, so stand-in snapshots carrying a `t` are enough.
times = (100.0, 101.0, 102.0)
window_vec = [(t = t,) for t in times]
window_tup = Tuple(window_vec)
cases = [   # time, expected pair (A, B) by index, expected weight on A
    (99.0,  (1, 2), 1.0),    # before the window: frozen at the first snapshot
    (100.25, (1, 2), 0.75),
    (101.0, (2, 3), 1.0),
    (101.5, (2, 3), 0.5),
    (102.0, (2, 3), 0.0),
    (104.97, (2, 3), 0.0),   # 2.97 M past the last snapshot, as at the crash: frozen at the last snapshot
]
for (name, data) in (("Vector (CPU)", window_vec), ("NTuple{3} (GPU)", window_tup))
    for (t, (a, b), w) in cases
        dA, dB, wA, interp = IH.set_tinterp_ns(X(t), model, data)
        @printf("%-16s t = %7.2f: pair (%.0f, %.0f), weight on A %.4f\n", name, t, dA.t, dB.t, wA)
        @assert interp "slow light must interpolate"
        @assert (dA.t, dB.t) == (times[a], times[b]) "t = $t: wrong pair ($(dA.t), $(dB.t))"
        @assert 0.0 <= wA <= 1.0 "t = $t: weight $wA outside [0, 1]: the flow would be extrapolated"
        @assert isapprox(wA, w; atol=1e-12) "t = $t: weight $wA, expected $w"
    end
end
println("\nPASSED: slow-light time interpolation never extrapolates")
