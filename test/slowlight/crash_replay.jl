# Replay of the M87 slow-light crash (Oct 19; investigated Oct 23): the render round of the snapshot window
# (01715, 01716, 01717) of the MAD run, with the M87 230 GHz camera of results_M87/slowlight_230GHz. Slow light
# evaluates the plasma at both ends of every step whose start lies in [t1, t2), out to the camera at r = 1000, and
# far out a step spans up to ~10 M of coordinate time, so its end can lie up to ~9 M past t3. With the time
# interpolation unclamped, pixel (408, 310) got theta_e = -1.0587e-3 at r = 459 (1/theta_e = -944.56, the argument
# the Bessel function rejected). Clamped (as ipole), every point past t3 sees the last snapshot.
# Asserts: the crash point now gets theta_e of the last snapshot, > 0; no step of the round gets theta_e below the
# 1e-3 floor; and, for the record, recomputes the unclamped value the crash saw.
# Usage: julia --project=scripts -t N crash_replay.jl DUMPDIR   (DUMPDIR holding torus.out0.01715..01717.phdf)
using Jipole, Printf, StaticArrays

dumpdir = ARGS[1]
dump(k) = joinpath(dumpdir, @sprintf("torus.out0.%05d.phdf", k))
all(isfile(dump(k)) for k in 1715:1717) || error("$dumpdir does not hold torus.out0.01715..01717.phdf")

# The M87 230 GHz slow-light run of Oct 19 (prep_M87.toml; EHT M87 Paper V values).
MBH, M_UNIT, RHIGH, THETA, FOV, D_PC, FREQ, RO, NPIX = 6.2e9, 6.934953355486971e24, 80.0, 163.0, 160.0, 16.9e6, 230e9, 1000.0, 420
T0 = 11185.1503   # camera time of its first frame (slow.00001.h5 frame/t_camera, = first dump time - tgeof)
reader = Jipole.Kharma
model = reader.read_header(dump(1715), MBH; th_beg=0.0174, Rlow=1.0, Rhigh=RHIGH, beta_crit=1.0, sigma_cut=1.0,
    sigma_cut_high=-1.0, M_unit=M_UNIT, slow_light=true)
Rh = 1 + sqrt(1.0 - model.a^2)
DX = D_PC * Jipole.Constants.PC / model.L_unit / Jipole.Constants.MUAS_PER_RAD * FOV
geos, nsteps, _, _, _ = Jipole.Imaging.trace_all_geodesics(model, RO, THETA, 0.0, FREQ, NPIX, NPIX, DX / RO, DX / RO,
    8000, Rh, 0.0, 0.0)
data = [reader.load_data(dump(k), RHIGH, model) for k in 1715:1717]
t1, t2, t3 = data[1].t, data[2].t, data[3].t

# The frame a step belongs to in this round: its start Xi = tau_i + T + 1e-5 lies in [t1, t2) (render_round_cpu!).
function step_in_round(traj, k)
    taui = traj[k].X[1]
    T = T0 + floor(Int, t2 - taui - 1e-5 - T0)
    Xi = taui + T + 1e-5
    return (t1 <= Xi < t2 && T >= T0) ? T : nothing
end
thetae_at(X) = (z = Jipole.Grid.locate(X, model); z.in_domain ? Jipole.Iharm.get_model_thetae(z, X, model, data) : NaN)

# 1. The crash point.
traj = geos[408, 310]
T = step_in_round(traj, 88)
T === nothing && error("step 88 of pixel (408, 310) is not in this round: the geodesics differ from the crash run")
Xf = SVector{4,Float64}(traj[87].X[1] + T + 1e-5, traj[87].X[2], traj[87].X[3], traj[87].X[4])
z = Jipole.Grid.locate(Xf, model)
th2, th3 = Jipole.Grid.interp_scalar(data[2].θe, z), Jipole.Grid.interp_scalar(data[3].θe, z)
w_raw = 1.0 - (Xf[1] - t2) / (t3 - t2)
unclamped = w_raw * th2 + (1 - w_raw) * th3
the = thetae_at(Xf)
@printf("crash point: pixel (408,310) step 88, frame T = %.2f, Xf = %.3f (%.3f M past t3)\n", T, Xf[1], Xf[1] - t3)
@printf("  theta_e at t2 %.4e, at t3 %.4e; unclamped extrapolation %.4e; now %.4e\n", th2, th3, unclamped, the)
@assert unclamped < 0 "the replay does not reach the crash state (unclamped theta_e $unclamped)"
@assert the > 0 && isapprox(the, th3; rtol=1e-12) "theta_e at the crash point is $the, expected the last snapshot's $th3"

# 2. Every step of the round: no theta_e below the floor anywhere slow light evaluates.
function round_min(geos, nsteps)
    lo = Inf; n = 0; npast = 0
    for j in axes(geos, 2), i in axes(geos, 1)
        traj = geos[i, j]
        for k in nsteps[i, j]:-1:3
            T = step_in_round(traj, k)
            T === nothing && continue
            for X0 in (traj[k].X, traj[k - 1].X)
                X = SVector{4,Float64}(X0[1] + T + 1e-5, X0[2], X0[3], X0[4])
                X[1] > t3 && (npast += 1)
                v = thetae_at(X)
                isnan(v) && continue
                lo = min(lo, v); n += 1
            end
        end
    end
    return lo, n, npast
end
lo, n, npast = round_min(geos, nsteps)
@printf("round: %d evaluations on the grid (%d past t3), smallest theta_e %.4e\n", n, npast, lo)
@assert npast > 0 "no step overshoots t3: the replay does not exercise the clamp"
@assert lo >= 1e-3 * (1 - 1e-12) "theta_e $lo below the 1e-3 floor: slow light still extrapolates somewhere"
println("\nPASSED: the M87 crash window no longer produces a negative theta_e")
