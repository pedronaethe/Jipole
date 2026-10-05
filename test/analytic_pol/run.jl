# Constant-coefficient polarized transfer test (arXiv:2303.12004, section 3.1; ipole's model/ldi2).
#
# Two calculations are written to the output file, for each of the two coefficient sets:
#
#  1. "solver": the Stokes-space solver Polarization.evolve_stokes marched on a uniform grid in
#     the affine parameter, to be compared with the closed-form solution.
#  2. "pipeline": the complete polarized transfer (parallel transport of the coherency tensor,
#     plasma tetrad, source step, camera projection) along a radial ray in flat space written in
#     spherical coordinates, exactly as ipole's ldi2 model sets it up, to be compared with ipole
#     step by step.
#
# Usage: julia --project=scripts run.jl OUTPUT_H5
using Jipole
using StaticArrays

const P = Jipole.Polarization
const HDF5 = Jipole.Output.HDF5
const C = Jipole.Constants

# Table 1 of arXiv:2303.12004 (ipole's ldi2_iq.par and ldi2_quv.par):
#                             jI   jQ   jU   jV   aI   aQ   aU   aV   rQ    rU   rV
const TESTS = (
    iq = P.PolCoeffs{Float64}(2.0, 1.0, 0.0, 0.0, 1.0, 1.2, 0.0, 0.0, 0.0, 0.0, 0.0),
    quv = P.PolCoeffs{Float64}(0.0, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, -4.0),
)
# Camera azimuth of the two ipole parameter files [deg]
const PHICAM = (iq = 90.0, quv = 180.0)

# ---------------------------------------------------------------------------------------------
# 1. Stokes-space solver on a uniform grid
# ---------------------------------------------------------------------------------------------

"""March `evolve_stokes` from zero over `nstep` steps of length `dl`; returns the history."""
function march_solver(c, dl, nstep)
    lam = zeros(nstep)
    S = zeros(4, nstep)
    s = (0.0, 0.0, 0.0, 0.0)
    for k in 1:nstep
        s = P.evolve_stokes(s..., c, dl)
        lam[k] = k * dl
        S[:, k] .= s
    end
    return lam, S
end

# ---------------------------------------------------------------------------------------------
# 2. Full pipeline in flat space: a test-local model, as ipole's model/ldi2
# ---------------------------------------------------------------------------------------------

"""
Flat space in spherical coordinates `X = (t, r, θ, φ)` filled with a medium of constant transfer
coefficients (ipole's `ldi2` model, `METRIC_MINKOWSKI`).
"""
struct Ldi2Params <: Jipole.AbstractModels.AbstractModel
    a::Float64
    metric::Int
    L_unit::Float64
    rmax_geo::Float64
    coeffs::P.PolCoeffs{Float64}
end

# ipole's INTEGRATOR_TEST uses the affine step as it is; this L_unit makes Jipole's conversion
# factor L_unit h / (m_e c^2) equal to one.
Ldi2Params(coeffs) = Ldi2Params(0.0, Jipole.Metrics.METRIC_MINKOWSKI, C.ME * C.CL * C.CL / C.HPL, 100.0, coeffs)

# Native coordinates are r and θ themselves.
Jipole.Coordinates.bl_coord(X, model::Ldi2Params, R0::Float64=0.0) = (X[2], X[3])

# Connection of flat space in spherical coordinates.
function Jipole.Geodesics.get_connection_analytic(X::AbstractVector{T}, bhspin, model::Ldi2Params) where {T}
    r, th = X[2], X[3]
    sth, cth = sin(th), cos(th)
    lconn = zeros(MArray{Tuple{4,4,4},T,3,64})
    lconn[2, 3, 3] = -r
    lconn[2, 4, 4] = -r * sth * sth
    lconn[3, 2, 3] = lconn[3, 3, 2] = 1 / r
    lconn[3, 4, 4] = -sth * cth
    lconn[4, 2, 4] = lconn[4, 4, 2] = 1 / r
    lconn[4, 3, 4] = lconn[4, 4, 3] = cth / sth
    return SArray(lconn)
end

Jipole.Radiation.radiating_region(X, model::Ldi2Params, Rh) = true

# Constant coefficients; a static observer; no magnetic field, so that evolve_n falls back to its
# guess B = (0, 1, 1, 1), as ipole's ldi2 model does.
function P.get_pol_state(X, Kcon, freq, bhspin, model::Ldi2Params, data)
    z = zero(SVector{4,Float64})
    return model.coeffs, SVector(1.0, 0.0, 0.0, 0.0), z, z
end

"""
Trace the ray of pixel (0, 0) of a 2x2 image backward from the camera to r = Rh, with the
geodesic integrator and step-size function of the main code, storing it as
`Geodesics.trace_geodesic` does. (That routine is not used because its stop test is written for
a logarithmic radial coordinate.) As in ipole's INTEGRATOR_TEST the wavevector is not scaled by
the frequency.
"""
function trace_ray(model, Xcam, fov, Rh)
    zero4 = zero(SVector{4,Float64})
    X = Xcam
    K = Jipole.Geodesics.init_kcon(0, 0, Xcam, 2, 2, fov, fov, model.a, model)
    traj = [Jipole.GeoTypes.OfTrajGeneric{Float64}(0.0, X, K, X, K)]
    while !(X[2] < Rh + 0.0001 || (X[2] > model.rmax_geo && K[2] < 0.0))
        dl = Jipole.Geodesics.stepsize(X, K, zero4, zero4)
        last = traj[end]
        traj[end] = Jipole.GeoTypes.OfTrajGeneric{Float64}(dl, last.X, last.Kcon, last.Xhalf, last.Kconhalf)
        X, K, Xhalf, Khalf = Jipole.Geodesics.push_photon(X, K, -dl, model.a, model)
        push!(traj, Jipole.GeoTypes.OfTrajGeneric{Float64}(0.0, X, K, Xhalf, Khalf))
    end
    return traj
end

"""
Run the transfer loop one step at a time, recording after each step the Stokes parameters in
the plasma frame (what ipole's ldi2 model records), then the camera-frame result.
"""
function run_pipeline(model, traj)
    nsteps = length(traj)
    Rh = 0.5
    dl_unit = model.L_unit * C.HPL / (C.ME * C.CL * C.CL)
    N = P.zero_tensor(Float64)
    tauF = 0.0
    lam = zeros(nsteps - 1)
    S = zeros(4, nsteps - 1)
    l = 0.0
    for (k, n) in enumerate(nsteps:-1:2)
        ti, tf = traj[n], traj[n-1]
        N, tauF = P.polarized_step(N, tauF, ti.X, ti.Kcon, ti.Xhalf, ti.Kconhalf, tf.X, tf.Kcon,
            tf.dl, dl_unit, Rh, 1.0, model.a, model, nothing)
        # Stokes parameters in the plasma tetrad at the end of the step
        gcov = Jipole.Metrics.gcov_func(tf.X, model.a, model)
        _, _, Ecov = P.plasma_tetrad(SVector(1.0, 0.0, 0.0, 0.0), tf.Kcon, SVector(0.0, 1.0, 1.0, 1.0), gcov)
        l += tf.dl
        lam[k] = l
        S[:, k] .= P.tensor_to_stokes(P.complex_coord_to_tetrad_rank2(N, Ecov))
    end
    # The same loop through the library routine, projected on the camera and stored as an image pixel.
    pol = zeros(P.NIMG, 1, 1)
    SI, SQ, SU, SV, tauF2 = P.integrate_emission_pol(traj, nsteps, 1.0, model.a, model, nothing)
    P.save_pixel!(pol, 1, 1, SI, SQ, SU, SV, tauF2, 1.0, 0)
    return lam, S, vec(pol)
end

# ---------------------------------------------------------------------------------------------

output_file = ARGS[1]
mkpath(dirname(output_file))
HDF5.h5open(output_file, "w") do f
    for name in keys(TESTS)
        c = TESTS[name]

        # The range 0 <= lambda <= 3 shown in the paper, at the step ipole takes (0.01) and at
        # half of it, to measure the order of convergence.
        for (label, nstep) in (("solver", 300), ("solver_half", 600))
            lam, S = march_solver(c, 3.0 / nstep, nstep)
            write(f, "$name/$label/lam", lam)
            write(f, "$name/$label/stokes", S)
        end

        model = Ldi2Params(c)
        Xcam = SVector(0.0, 10.0, π / 2, PHICAM[name] / 180 * π)
        traj = trace_ray(model, Xcam, 1.0e-10 / 10.0, 0.5)
        lam, S, pol = run_pipeline(model, traj)
        write(f, "$name/pipeline/lam", lam)
        write(f, "$name/pipeline/stokes", S)
        write(f, "$name/pipeline/pol", pol)
        println("$name: pipeline took $(length(traj) - 1) steps to lambda = $(lam[end])")
    end
end
println("Wrote $output_file")
