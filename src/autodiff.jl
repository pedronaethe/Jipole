"""
Differentiable ray tracing: computes the image intensity together with
its derivatives with respect to observer inclination (`θo`) and either
black hole spin `a` (`Analytic`/`ThinDisk`) or `Rhigh`
(`Iharm`), via `ForwardDiff`.
"""
module Autodiff

using LinearAlgebra
using StaticArrays
using ForwardDiff
using ..Constants
using ..GeoTypes
using ..Coordinates
using ..Camera
using ..Geodesics
using ..Radiation
using ..DebugFunctions
using ..ThinDisk

export AutoDiffGeoTrajEulerMethod!, AutoDiffGeoTrajEulerMethod_GRMHD!

"""
    Mom4ODE(X, Kcon, bhspin, model)

Right-hand side of the photon geodesic ODE, `dK^μ/dλ = -Γ^μ_{αβ} K^α K^β`.

# Arguments
- `X`: Position four-vector in internal coordinates.
- `Kcon`: Contravariant photon 4-momentum.
- `bhspin`: Dimensionless black hole spin parameter.
- `model`: Model parameters.

# Returns
- `dKcon/dλ`.
"""
function Mom4ODE(X::AbstractVector, Kcon::AbstractVector, bhspin, model)
    T = eltype(Kcon)

    lconn = Geodesics.get_connection_analytic(X, bhspin, model)

    result = zero(MVector{4,T})

    @inbounds for mu in 1:4
        for alpha in 1:4
            for beta in 1:4
                result[mu] -= lconn[mu, alpha, beta] * Kcon[alpha] * Kcon[beta]
            end
        end
    end
    return SVector(result)
end

"""
    CalculateK(ro, θo, phi, i, j, nx, ny, fovx, fovy, bhspin, freq, model)

Compute the unitless photon 4-momentum launched from camera pixel
`(i, j)`.

# Arguments
- `ro`, `θo`, `phi`: Camera radial distance, inclination, and azimuth.
- `i`, `j`: Pixel indices in the image plane.
- `nx`, `ny`: Image resolution.
- `fovx`, `fovy`: Field of view, in radians.
- `bhspin`: Dimensionless black hole spin parameter.
- `freq`: Frequency, in cgs units.
- `model`: Model parameters.

# Returns
- The unitless contravariant photon 4-momentum.
"""
function CalculateK(ro, θo, phi, i, j, nx, ny, fovx, fovy, bhspin, freq, model)
    Xcam = Camera.camera_position(ro, θo, phi, bhspin, model)
    T = eltype(Xcam)
    Kcon = MVector{4,T}(undef)
    X = MVector{4,T}(undef)
    Geodesics.init_XK!(X, Kcon, i, j, Xcam, nx, ny, fovx, fovy, bhspin, model)
    return SVector(Kcon) * (freq * Constants.HPL / (Constants.ME * Constants.CL * Constants.CL))
end

"""
    RadTransferDiff(Xi, Kconi, freq, Ii, bhspin, model, data)

Right-hand side of the (linearized) radiative transfer equation at a
single point, `dI/dλ = j - k I`.

# Arguments
- `Xi`: Position four-vector in internal coordinates.
- `Kconi`: Contravariant photon 4-momentum.
- `freq`: Frequency, in cgs units.
- `Ii`: Current intensity.
- `bhspin`: Dimensionless black hole spin parameter.
- `model`: Model parameters.
- `data`: Model-specific auxiliary data.

# Returns
- `dI/dλ`.
"""
function RadTransferDiff(Xi, Kconi, freq, Ii, bhspin, model, data)
    ji, ki = Radiation.get_jk(Xi, Kconi, freq, bhspin, model, data, Val(false))
    return ji - ki * Ii
end

"""
    transfer_step(I_prev, X_curr, K_curr, X_next, K_next, dl, freq, bhspin, model, data)

Evolve the intensity over one geodesic segment, recomputing `j`/`k` at
both endpoints (used by [`ApproxSolveWrapper`](@ref)-style autodiff
wrappers).

# Arguments
- `I_prev`: Intensity at the start of the segment.
- `X_curr`, `K_curr`: Position/4-momentum at the start of the segment.
- `X_next`, `K_next`: Position/4-momentum at the end of the segment.
- `dl`: Length of the segment along the geodesic.
- `freq`: Frequency, in cgs units.
- `bhspin`: Dimensionless black hole spin parameter.
- `model`: Model parameters.
- `data`: Model-specific auxiliary data.

# Returns
- The intensity at the end of the segment.
"""
function transfer_step(I_prev, X_curr, K_curr, X_next, K_next, dl, freq, bhspin, model, data)
    ji, ki, _, _ = Radiation.get_jk(X_curr, K_curr, freq, bhspin, model, data, Val(false))
    jf, kf, _, _ = Radiation.get_jk(X_next, K_next, freq, bhspin, model, data, Val(false))

    return Radiation.approximate_solve(I_prev, ji, ki, jf, kf, dl)
end

"""
    AutoDiffGeoTrajEulerMethod!(traj, dI_dθo_out, intensity_out, dI_da_out, ro, θo, phi, bhspin, nx, ny, nmaxstep, i, j, freq, fovx, fovy, model, Rstop, data=nothing)

Compute the intensity and its derivatives with respect to `θo` and `a`
for pixel `(i, j)`, using forward-mode autodiff through the geodesic
integration (`Analytic`/`ThinDisk` models only — `Iharm` uses
[`AutoDiffGeoTrajEulerMethod_GRMHD!`](@ref)).

# Arguments
- `traj`: Scratch trajectory vector, emptied by this function.
- `dI_dθo_out`, `intensity_out`, `dI_da_out`: Output references.
- `ro`, `θo`, `phi`: Camera radial distance, inclination, and azimuth.
- `bhspin`: Dimensionless black hole spin parameter.
- `nx`, `ny`: Image resolution.
- `nmaxstep`: Maximum number of integration steps.
- `i`, `j`: Pixel indices in the image plane.
- `freq`: Frequency, in cgs units.
- `fovx`, `fovy`: Field of view, in radians.
- `model`: Model parameters.
- `Rstop`: Backward-integration stopping radius.
- `data`: Model-specific auxiliary data.
"""
function AutoDiffGeoTrajEulerMethod!(traj, dI_dθo_out::Base.RefValue{Float64}, intensity_out::Base.RefValue{Float64}, dI_da_out::Base.RefValue{Float64}, ro::Float64, θo::Float64, phi::Float64, bhspin::Float64, nx::Int64, ny::Int64, nmaxstep::Int64, i::Int64, j::Int64, freq::Float64, fovx::Float64, fovy::Float64, model, Rstop::Float64, data=nothing)
    Xcam = MVector{4,Float64}(Camera.camera_position(ro, θo, phi, bhspin, model))
    Kcon = MVector{4,Float64}(undef)
    X = MVector{4,Float64}(undef)
    Rh = 1 + sqrt(1.0 - bhspin * bhspin)

    Geodesics.init_XK!(X, Kcon, i, j, Xcam, nx, ny, fovx, fovy, bhspin, model)
    Kcon .*= freq * Constants.HPL / (Constants.ME * Constants.CL * Constants.CL)
    dl_unit::Float64 = model.L_unit * Constants.HPL / (Constants.ME * Constants.CL^2)

    Xhalf = copy(X)
    Kconhalf = copy(Kcon)
    lconn = MArray{Tuple{4,4,4},Float64,3,64}(undef)

    jac = MMatrix{4,9,Float64}(undef)
    dX_dθo = ForwardDiff.derivative(x -> Camera.camera_position(ro, x, phi, bhspin, model), θo)
    dK_dθo = ForwardDiff.derivative(x -> CalculateK(ro, x, phi, i, j, nx, ny, fovx, fovy, bhspin, freq, model), θo)

    dX_da = ForwardDiff.derivative(x -> Camera.camera_position(ro, θo, phi, x, model), bhspin)
    dK_da = ForwardDiff.derivative(x -> CalculateK(ro, θo, phi, i, j, nx, ny, fovx, fovy, x, freq, model), bhspin)

    systemODEs_flat = XK -> begin
        Xs = SVector{4}(XK[1], XK[2], XK[3], XK[4])
        Ks = SVector{4}(XK[5], XK[6], XK[7], XK[8])
        spin = XK[9]
        Mom4ODE(Xs, Ks, spin, model)
    end

    XK = MVector{9,Float64}(undef)
    XK[9] = bhspin

    push!(traj, OfTraj(
        0.0,
        SVector{4,Float64}(X),
        SVector{4,Float64}(Kcon),
        SVector{4,Float64}(Xhalf),
        SVector{4,Float64}(Kconhalf),
        SVector{4,Float64}(dX_dθo),
        SVector{4,Float64}(dK_dθo),
        SVector{4,Float64}(dX_da),
        SVector{4,Float64}(dK_da)
    ))

    step::Int64 = 1

    temp_dX_dθo = MVector{4,Float64}(undef)
    temp_dX_da = MVector{4,Float64}(undef)
    temp_dK_dθo = MVector{4,Float64}(undef)
    temp_dK_da = MVector{4,Float64}(undef)

    temp_jac_dX_dθo = MVector{4,Float64}(undef)
    temp_jac_dK_dθo = MVector{4,Float64}(undef)
    temp_jac_dX_da = MVector{4,Float64}(undef)
    temp_jac_dK_da = MVector{4,Float64}(undef)
    cstartx = MVector{4,Float64}(0.0, log(Rh), 0.0, 0.0)
    cstopx = MVector{4,Float64}(0.0, log(model.Rout), 1.0, 2.0 * π)
    while (Geodesics.stop_backward_integration(X, Kcon, Rh, Rstop) == 0 && (step <= nmaxstep))
        @inbounds begin
            @inbounds for k = 1:4
                XK[k] = X[k]
                XK[k+4] = Kcon[k]
            end

            ForwardDiff.jacobian!(jac, systemODEs_flat, XK)

            dl = Geodesics.stepsize(X, Kcon, cstartx, cstopx)
            scaled_dl = dl * dl_unit

            @. temp_dX_dθo = traj[step].dX_dθo - dl * traj[step].dK_dθo
            @. temp_dX_da = traj[step].dX_da - dl * traj[step].dK_da

            mul!(temp_jac_dX_dθo, view(jac, 1:4, 1:4), traj[step].dX_dθo)
            mul!(temp_jac_dK_dθo, view(jac, 1:4, 5:8), traj[step].dK_dθo)
            @. temp_dK_dθo = traj[step].dK_dθo - dl * (temp_jac_dX_dθo + temp_jac_dK_dθo)

            mul!(temp_jac_dX_da, view(jac, 1:4, 1:4), traj[step].dX_da)
            mul!(temp_jac_dK_da, view(jac, 1:4, 5:8), traj[step].dK_da)
            @. temp_dK_da = traj[step].dK_da - dl * (temp_jac_dX_da + temp_jac_dK_da)

            for k in 1:4
                temp_dK_da[k] = temp_dK_da[k] - dl * jac[k, 9]
            end

            Geodesics.push_photon!(X, Kcon, -dl, Xhalf, Kconhalf, lconn, bhspin, model)

            step += 1
            push!(traj, OfTraj(
                scaled_dl,
                SVector{4,Float64}(X),
                SVector{4,Float64}(Kcon),
                SVector{4,Float64}(Xhalf),
                SVector{4,Float64}(Kconhalf),
                SVector{4,Float64}(temp_dX_dθo),
                SVector{4,Float64}(temp_dK_dθo),
                SVector{4,Float64}(temp_dX_da),
                SVector{4,Float64}(temp_dK_da)
            ))
        end
    end

    if step > nmaxstep
        @error("AutoDiffGeoTrajEulerMethod: Maximum number of steps reached without meeting geodesics stop condition.")
        error()
    end

    Intensity = 0.0
    dI_dθo = 0.0
    dI_da = 0.0
    jac_I_X = MVector{4,Float64}(undef)
    jac_I_K = MVector{4,Float64}(undef)

    Xi_S = traj[step].X
    Kconi_S = traj[step].Kcon

    ji, ki = Radiation.get_jk(Xi_S, Kconi_S, freq, bhspin, model, data, Val(false))

    for nstep = step:-1:2
        Xi_S = traj[nstep].X
        Xf_S = traj[nstep-1].X
        Kconi_S = traj[nstep].Kcon
        Kconf_S = traj[nstep-1].Kcon

        if model isa ThinDisk.ThinDiskParams
            if ThinDisk.thindisk_region(Xi_S, Xf_S, model)
                Intensity = ThinDisk.GetTDBoundaryCondition(Xi_S, Kconi_S, bhspin, Rh, model)
            end
            continue
        end

        if !Radiation.radiating_region(Xf_S, model, Rh)
            continue
        end

        rad_x = x -> RadTransferDiff(x, Kconi_S, freq, Intensity, bhspin, model, data)
        rad_k = k -> RadTransferDiff(Xi_S, k, freq, Intensity, bhspin, model, data)
        rad_a = spin -> RadTransferDiff(Xi_S, Kconi_S, freq, Intensity, spin, model, data)
        rad_i = intens -> RadTransferDiff(Xi_S, Kconi_S, freq, intens, bhspin, model, data)

        ForwardDiff.gradient!(jac_I_X, rad_x, Xi_S)
        ForwardDiff.gradient!(jac_I_K, rad_k, Kconi_S)
        jac_I_A = ForwardDiff.derivative(rad_a, bhspin)
        jac_I_I = ForwardDiff.derivative(rad_i, Intensity)

        dI_dθo = dI_dθo + (traj[nstep].dl) * (dot(jac_I_X, traj[nstep].dX_dθo) + dot(jac_I_K, traj[nstep].dK_dθo) + jac_I_I * dI_dθo)
        dI_da = dI_da + (traj[nstep].dl) * (dot(jac_I_X, traj[nstep].dX_da) + dot(jac_I_K, traj[nstep].dK_da) + jac_I_I * dI_da + jac_I_A)

        jf, kf = Radiation.get_jk(Xf_S, Kconf_S, freq, bhspin, model, data, Val(false))
        Intensity = Radiation.approximate_solve(Intensity, ji, ki, jf, kf, traj[nstep-1].dl)
        if isnan(Intensity) || isinf(Intensity)
            @error "NaN or Inf encountered in intensity calculation at pixel ($i, $j)"
            println("Intensity = $Intensity, ji = $ji, ki = $ki, jf = $jf, kf = $kf")
            DebugFunctions.print_vector("Kconf =", Kconf_S)
            DebugFunctions.print_vector("Kconi =", Kconi_S)
            error("NaN or Inf encountered in intensity calculation")
        end

        ji = jf
        ki = kf
    end

    dI_dθo_out[] = dI_dθo * freq^3
    intensity_out[] = Intensity * freq^3
    dI_da_out[] = dI_da * freq^3
    empty!(traj)
    return nothing
end

"""
    AutoDiffGeoTrajEulerMethod_GRMHD!(traj, dI_dθo_out, intensity_out, dI_dRhigh_out, ro, θo, phi, bhspin, nx, ny, nmaxstep, i, j, freq, fovx, fovy, model, Rstop, data=nothing)

Compute the intensity and its derivatives with respect to `θo` and
`Rhigh` for pixel `(i, j)`, using forward-mode autodiff through the
geodesic integration for `θo`, and the analytic `dj/dRhigh`/`dk/dRhigh`
from [`Iharm.jar_calc`](@ref) for `Rhigh`.

# Arguments
- `traj`: Scratch trajectory vector, emptied by this function.
- `dI_dθo_out`, `intensity_out`, `dI_dRhigh_out`: Output references.
- `ro`, `θo`, `phi`: Camera radial distance, inclination, and azimuth.
- `bhspin`: Dimensionless black hole spin parameter.
- `nx`, `ny`: Image resolution.
- `nmaxstep`: Maximum number of integration steps.
- `i`, `j`: Pixel indices in the image plane.
- `freq`: Frequency, in cgs units.
- `fovx`, `fovy`: Field of view, in radians.
- `model`: Iharm model parameters.
- `Rstop`: Backward-integration stopping radius.
- `data`: GRMHD snapshot(s), already loaded with the `Rhigh` at which the
  derivative is requested.
"""
function AutoDiffGeoTrajEulerMethod_GRMHD!(traj, dI_dθo_out::Base.RefValue{Float64}, intensity_out::Base.RefValue{Float64}, dI_dRhigh_out::Base.RefValue{Float64}, ro::Float64, θo::Float64, phi::Float64, bhspin::Float64, nx::Int64, ny::Int64, nmaxstep::Int64, i::Int64, j::Int64, freq::Float64, fovx::Float64, fovy::Float64, model, Rstop::Float64, data::T_data=nothing) where {T_data}
    empty!(traj)

    Xcam = MVector{4,Float64}(Camera.camera_position(ro, θo, phi, bhspin, model))

    Kcon = MVector{4,Float64}(undef)
    X = MVector{4,Float64}(undef)
    Rh = 1 + sqrt(1.0 - bhspin * bhspin)

    Geodesics.init_XK!(X, Kcon, i, j, Xcam, nx, ny, fovx, fovy, bhspin, model)
    Kcon .*= freq * Constants.HPL / (Constants.ME * Constants.CL * Constants.CL)
    dl_unit::Float64 = model.L_unit * Constants.HPL / (Constants.ME * Constants.CL^2)

    Xhalf = copy(X)
    Kconhalf = copy(Kcon)
    lconn = MArray{Tuple{4,4,4},Float64,3,64}(undef)

    jac = MMatrix{4,9,Float64}(undef)
    dX_dθo = ForwardDiff.derivative(x -> Camera.camera_position(ro, x, phi, bhspin, model), θo)
    dK_dθo = ForwardDiff.derivative(x -> CalculateK(ro, x, phi, i, j, nx, ny, fovx, fovy, bhspin, freq, model), θo)
    XK = MVector{9,Float64}(undef)
    XK[9] = bhspin

    systemODEs_flat = xk -> begin
        Xs = SVector{4}(xk[1], xk[2], xk[3], xk[4])
        Ks = SVector{4}(xk[5], xk[6], xk[7], xk[8])
        spin = xk[9]
        Mom4ODE(Xs, Ks, spin, model)
    end

    push!(traj, OfTraj(
        0.0,
        SVector{4,Float64}(X),
        SVector{4,Float64}(Kcon),
        SVector{4,Float64}(Xhalf),
        SVector{4,Float64}(Kconhalf),
        SVector{4,Float64}(dX_dθo),
        SVector{4,Float64}(dK_dθo),
        zero(SVector{4,Float64}),
        zero(SVector{4,Float64})
    ))

    step::Int64 = 1

    temp_dX_dθo = MVector{4,Float64}(undef)
    temp_dK_dθo = MVector{4,Float64}(undef)
    temp_jac_dX_dθo = MVector{4,Float64}(undef)
    temp_jac_dK_dθo = MVector{4,Float64}(undef)

    while (Geodesics.stop_backward_integration(X, Kcon, Rh, Rstop) == 0 && (step <= nmaxstep))
        @inbounds begin
            @inbounds for k = 1:4
                XK[k] = X[k]
                XK[k+4] = Kcon[k]
            end

            jac_static = ForwardDiff.jacobian(systemODEs_flat, SVector(XK))
            jac .= jac_static

            dl = Geodesics.stepsize(X, Kcon, model.cstartx, model.cstopx)

            scaled_dl = dl * dl_unit

            @. temp_dX_dθo = traj[step].dX_dθo - dl * traj[step].dK_dθo

            mul!(temp_jac_dX_dθo, view(jac, 1:4, 1:4), traj[step].dX_dθo)
            mul!(temp_jac_dK_dθo, view(jac, 1:4, 5:8), traj[step].dK_dθo)
            @. temp_dK_dθo = traj[step].dK_dθo - dl * (temp_jac_dX_dθo + temp_jac_dK_dθo)

            Geodesics.push_photon!(X, Kcon, -dl, Xhalf, Kconhalf, lconn, bhspin, model)

            step += 1
            push!(traj, OfTraj(
                scaled_dl,
                SVector{4,Float64}(X),
                SVector{4,Float64}(Kcon),
                SVector{4,Float64}(Xhalf),
                SVector{4,Float64}(Kconhalf),
                SVector{4,Float64}(temp_dX_dθo),
                SVector{4,Float64}(temp_dK_dθo),
                zero(SVector{4,Float64}),
                zero(SVector{4,Float64})
            ))
        end
    end

    if step > nmaxstep
        @error("AutoDiffGeoTrajEulerMethod: Maximum number of steps reached without meeting geodesics stop condition.")
        error()
    end

    Intensity = 0.0
    dI_dθo = 0.0
    dI_dRhigh = 0.0

    Xi_S = traj[step].X
    Kconi_S = traj[step].Kcon

    ji, ki, dji_dRhigh, dki_dRhigh = Radiation.get_jk(Xi_S, Kconi_S, freq, bhspin, model, data, Val(true))

    for nstep = step:-1:2
        Xi_S = traj[nstep].X
        Xf_S = traj[nstep-1].X
        Kconi_S = traj[nstep].Kcon
        Kconf_S = traj[nstep-1].Kcon

        if !Radiation.radiating_region(Xf_S, model, Rh)
            continue
        end

        dl_step = traj[nstep-1].dl
        local_I = Intensity

        tw0 = I_new -> transfer_step(I_new, Xi_S, Kconi_S, Xf_S, Kconf_S, dl_step, freq, bhspin, model, data)
        tw1 = Xi_new -> transfer_step(local_I, Xi_new, Kconi_S, Xf_S, Kconf_S, dl_step, freq, bhspin, model, data)
        tw2 = Ki_new -> transfer_step(local_I, Xi_S, Ki_new, Xf_S, Kconf_S, dl_step, freq, bhspin, model, data)
        tw3 = Xf_new -> transfer_step(local_I, Xi_S, Kconi_S, Xf_new, Kconf_S, dl_step, freq, bhspin, model, data)
        tw4 = Kf_new -> transfer_step(local_I, Xi_S, Kconi_S, Xf_S, Kf_new, dl_step, freq, bhspin, model, data)

        jac_I_I = ForwardDiff.derivative(tw0, local_I)
        jac_I_Xi = ForwardDiff.gradient(tw1, Xi_S)
        jac_I_Ki = ForwardDiff.gradient(tw2, Kconi_S)
        jac_I_Xf = ForwardDiff.gradient(tw3, Xf_S)
        jac_I_Kf = ForwardDiff.gradient(tw4, Kconf_S)

        term_geom_i = dot(jac_I_Xi, traj[nstep].dX_dθo) + dot(jac_I_Ki, traj[nstep].dK_dθo)
        term_geom_f = dot(jac_I_Xf, traj[nstep-1].dX_dθo) + dot(jac_I_Kf, traj[nstep-1].dK_dθo)

        dI_dθo = (jac_I_I * dI_dθo) + term_geom_i + term_geom_f
        jf, kf, djf_dRhigh, dkf_dRhigh = Radiation.get_jk(Xf_S, Kconf_S, freq, bhspin, model, data, Val(true))

        internal_grads = ForwardDiff.gradient(v -> Radiation.approximate_solve(local_I, v[1], v[2], v[3], v[4], dl_step), SVector(ji, ki, jf, kf))

        dI_dji_solve, dI_dki_solve, dI_djf_solve, dI_dkf_solve = internal_grads
        dI_dRhigh = (jac_I_I * dI_dRhigh) + (dI_dji_solve * dji_dRhigh) + (dI_dki_solve * dki_dRhigh) + (dI_djf_solve * djf_dRhigh) + (dI_dkf_solve * dkf_dRhigh)

        Intensity = Radiation.approximate_solve(Intensity, ji, ki, jf, kf, dl_step)

        if isnan(Intensity) || isinf(Intensity)
            @error "NaN or Inf encountered in intensity calculation at pixel ($i, $j)"
            println("Intensity = $Intensity, ji = $ji, ki = $ki, jf = $jf, kf = $kf")
            DebugFunctions.print_vector("Kconf =", Kconf_S)
            DebugFunctions.print_vector("Kconi =", Kconi_S)
            error("NaN or Inf encountered in intensity calculation")
        end

        ji = jf
        ki = kf
        dji_dRhigh = djf_dRhigh
        dki_dRhigh = dkf_dRhigh
    end

    dI_dθo_out[] = dI_dθo * freq^3
    dI_dRhigh_out[] = dI_dRhigh * freq^3
    intensity_out[] = Intensity * freq^3
    empty!(traj)
    return nothing
end

end
