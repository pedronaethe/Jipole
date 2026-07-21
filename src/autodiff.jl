"""
Differentiable ray tracing: computes the image intensity together with
its derivatives with respect to observer inclination (`θo`) and either
black hole spin `a` (`Analytic`/`ThinDisk`) or `Rhigh`
(`Iharm`), via `ForwardDiff`. We plan to extend this to more parameters as a futured developments.
"""
module Autodiff

using LinearAlgebra
using StaticArrays
using ForwardDiff
using CUDA
using ..Iharm
using ..Constants
using ..GeoTypes
using ..Coordinates
using ..Camera
using ..Geodesics
using ..Radiation
using ..DebugFunctions
using ..ThinDisk
using ..Imaging
export autodiff_geo_traj_euler_method!, autodiff_geo_traj_euler_method_grmhd!, raytrace_gradients_gpu!, calculate_gradients, differentiate_pixel_intensity

"""
    momentum_ode(X, Kcon, bhspin, model)

Right-hand side of the photon geodesic ODE, `dK^μ/dλ = -Γ^μ_{αβ} K^α K^β`.

# Arguments
- `X`: Position four-vector in internal coordinates.
- `Kcon`: Contravariant photon 4-momentum.
- `bhspin`: Dimensionless black hole spin parameter.
- `model`: Model parameters.

# Returns
- `dKcon/dλ`.
"""
function momentum_ode(X::AbstractVector, Kcon::AbstractVector, bhspin, model)
    T = promote_type(eltype(X), eltype(Kcon))

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
    calculate_kcon(ro, θo, phi, i, j, nx, ny, fovx, fovy, bhspin, freq, model)

Compute the unitless photon 4-momentum launched from camera pixel
`(i, j)`. 

A tetrad is constructed at the camera position, and the photon 4-momentum is computed in that tetrad frame, then normalized and transformed to the internal coordinate system.

# Arguments
- `ro`, `θo`, `phi`: Camera radial distance, inclination, and azimuth.
- `i`, `j`: Pixel indices in the image plane.
- `nx`, `ny`: Image resolution.
- `fovx`, `fovy`: Field of view.
- `bhspin`: Dimensionless black hole spin parameter.
- `freq`: Frequency, in cgs units.
- `model`: Model parameters.

# Returns
- The unitless contravariant photon 4-momentum in the internal coordinate system.
"""
function calculate_kcon(ro, θo, phi, i, j, nx, ny, fovx, fovy, bhspin, freq, model)
    Xcam = Camera.camera_position(ro, θo, phi, bhspin, model)
    T = eltype(Xcam)
    Kcon = MVector{4,T}(undef)
    X = MVector{4,T}(undef)
    Geodesics.init_xk!(X, Kcon, i, j, Xcam, nx, ny, fovx, fovy, bhspin, model)
    return SVector(Kcon) * (freq * Constants.HPL / (Constants.ME * Constants.CL * Constants.CL))
end

"""
    rad_transfer_diff(Xi, Kconi, freq, Ii, bhspin, model, data)

Right-hand side of the (linearized) radiative transfer equation at a
single point, `dI/dλ = j - k I`. This function is used solely for computing the derivatives for the analytical model.

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
function rad_transfer_diff(Xi, Kconi, freq, Ii, bhspin, model, data)
    ji, ki = Radiation.get_jk(Xi, Kconi, freq, bhspin, model, data, Val(false))
    return ji - ki * Ii
end

"""
    transfer_step(I_prev, X_curr, K_curr, X_next, K_next, dl, freq, bhspin, model, data)

Evolve the intensity over one geodesic segment, recomputing `j`/`k` at
both endpoints (used by [`ApproxSolveWrapper`](@ref)-style autodiff
wrappers).

Intensity is calculated as following:
I = I_prev * exp(-k_avg * dl) + j_avg / k_avg * (1 - exp(-k_avg * dl))

where j_avg = (j_curr + j_next) / 2 and k_avg = (k_curr + k_next) / 2.

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
    autodiff_geo_traj_euler_method!(traj, dI_dθo_out, intensity_out, dI_da_out, ro, θo, phi, bhspin, nx, ny, nmaxstep, i, j, freq, fovx, fovy, model, Rstop, data=nothing)

Compute the intensity and its derivatives with respect to `θo` and `a`
for pixel `(i, j)`, using forward-mode autodiff through the geodesic
integration (`Analytic`/`ThinDisk` models only — `Iharm` uses
[`autodiff_geo_traj_euler_method_grmhd!`](@ref)).

This follows the equations defined in [Naethe Motta et al. 2025](https://iopscience.iop.org/article/10.3847/1538-4357/ae16a0)

# Arguments
- `traj`: Scratch trajectory vector, emptied by this function.
- `dI_dθo_out`, `intensity_out`, `dI_da_out`: Output references.
- `ro`, `θo`, `phi`: Camera radial distance, inclination, and azimuth.
- `bhspin`: Dimensionless black hole spin parameter.
- `nx`, `ny`: Image resolution.
- `nmaxstep`: Maximum number of integration steps.
- `i`, `j`: Pixel indices in the image plane.
- `freq`: Frequency, in cgs units.
- `fovx`, `fovy`: Field of view.
- `model`: Model parameters.
- `Rstop`: Backward-integration stopping radius.
- `data`: Model-specific auxiliary data. It is not used for the `Analytic`/`ThinDisk` models. Should be taken out in the future.
"""
function autodiff_geo_traj_euler_method!(traj, dI_dθo_out::Base.RefValue{Float64}, intensity_out::Base.RefValue{Float64}, dI_da_out::Base.RefValue{Float64}, ro::Float64, θo::Float64, phi::Float64, bhspin::Float64, nx::Int64, ny::Int64, nmaxstep::Int64, i::Int64, j::Int64, freq::Float64, fovx::Float64, fovy::Float64, model, Rstop::Float64, data=nothing)
    #TODO: Remove data argument in the future, as it is not used for the Analytic/ThinDisk models. It is only used for the Iharm model, which has its own autodiff function.
    Xcam = MVector{4,Float64}(Camera.camera_position(ro, θo, phi, bhspin, model))
    Kcon = MVector{4,Float64}(undef)
    X = MVector{4,Float64}(undef)
    Rh = 1 + sqrt(1.0 - bhspin * bhspin)

    Geodesics.init_xk!(X, Kcon, i, j, Xcam, nx, ny, fovx, fovy, bhspin, model)
    Kcon .*= freq * Constants.HPL / (Constants.ME * Constants.CL * Constants.CL)
    dl_unit::Float64 = model.L_unit * Constants.HPL / (Constants.ME * Constants.CL^2)

    Xhalf = copy(X)
    Kconhalf = copy(Kcon)
    lconn = MArray{Tuple{4,4,4},Float64,3,64}(undef)

    jac = MMatrix{4,9,Float64}(undef)
    dX_dθo = ForwardDiff.derivative(x -> Camera.camera_position(ro, x, phi, bhspin, model), θo)
    dK_dθo = ForwardDiff.derivative(x -> calculate_kcon(ro, x, phi, i, j, nx, ny, fovx, fovy, bhspin, freq, model), θo)

    dX_da = ForwardDiff.derivative(x -> Camera.camera_position(ro, θo, phi, x, model), bhspin)
    dK_da = ForwardDiff.derivative(x -> calculate_kcon(ro, θo, phi, i, j, nx, ny, fovx, fovy, x, freq, model), bhspin)

    systemODEs_flat = XK -> begin
        Xs = SVector{4}(XK[1], XK[2], XK[3], XK[4])
        Ks = SVector{4}(XK[5], XK[6], XK[7], XK[8])
        spin = XK[9]
        momentum_ode(Xs, Ks, spin, model)
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
                Intensity = ThinDisk.get_td_boundary_condition(Xi_S, Kconi_S, bhspin, Rh, model)
            end
            continue
        end

        if !Radiation.radiating_region(Xf_S, model, Rh)
            continue
        end

        rad_x = x -> rad_transfer_diff(x, Kconi_S, freq, Intensity, bhspin, model, data)
        rad_k = k -> rad_transfer_diff(Xi_S, k, freq, Intensity, bhspin, model, data)
        rad_a = spin -> rad_transfer_diff(Xi_S, Kconi_S, freq, Intensity, spin, model, data)
        rad_i = intens -> rad_transfer_diff(Xi_S, Kconi_S, freq, intens, bhspin, model, data)

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
    autodiff_geo_traj_euler_method_grmhd!(traj, dI_dθo_out, intensity_out, dI_dRhigh_out, ro, θo, phi, bhspin, nx, ny, nmaxstep, i, j, freq, fovx, fovy, model, Rstop, data=nothing)

Compute the intensity and its derivatives with respect to `θo` and
`Rhigh` for pixel `(i, j)`, using forward-mode autodiff through the
geodesic integration for `θo`, and the analytic `dj/dRhigh`/`dk/dRhigh`
from [`Iharm.jar_calc`](@ref) for `Rhigh`.

    This is done as discussed in [Naethe Motta et al. 2026] (https://iopscience.iop.org/article/10.3847/1538-4357/ae733f)

# Arguments
- `traj`: Scratch trajectory vector, emptied by this function.
- `dI_dθo_out`, `intensity_out`, `dI_dRhigh_out`: Output references.
- `ro`, `θo`, `phi`: Camera radial distance, inclination, and azimuth.
- `bhspin`: Dimensionless black hole spin parameter.
- `nx`, `ny`: Image resolution.
- `nmaxstep`: Maximum number of integration steps.
- `i`, `j`: Pixel indices in the image plane.
- `freq`: Frequency, in cgs units.
- `fovx`, `fovy`: Field of view.
- `model`: Iharm model parameters.
- `Rstop`: Backward-integration stopping radius.
- `data`: GRMHD snapshot(s), already loaded with the `Rhigh` at which the
  derivative is requested.
"""
function autodiff_geo_traj_euler_method_grmhd!(traj, dI_dθo_out::Base.RefValue{Float64}, intensity_out::Base.RefValue{Float64}, dI_dRhigh_out::Base.RefValue{Float64}, ro::Float64, θo::Float64, phi::Float64, bhspin::Float64, nx::Int64, ny::Int64, nmaxstep::Int64, i::Int64, j::Int64, freq::Float64, fovx::Float64, fovy::Float64, model, Rstop::Float64, data::T_data=nothing) where {T_data}
    empty!(traj)

    Xcam = MVector{4,Float64}(Camera.camera_position(ro, θo, phi, bhspin, model))

    Kcon = MVector{4,Float64}(undef)
    X = MVector{4,Float64}(undef)
    Rh = 1 + sqrt(1.0 - bhspin * bhspin)

    Geodesics.init_xk!(X, Kcon, i, j, Xcam, nx, ny, fovx, fovy, bhspin, model)
    Kcon .*= freq * Constants.HPL / (Constants.ME * Constants.CL * Constants.CL)
    dl_unit::Float64 = model.L_unit * Constants.HPL / (Constants.ME * Constants.CL^2)

    Xhalf = copy(X)
    Kconhalf = copy(Kcon)
    lconn = MArray{Tuple{4,4,4},Float64,3,64}(undef)

    jac = MMatrix{4,9,Float64}(undef)
    dX_dθo = ForwardDiff.derivative(x -> Camera.camera_position(ro, x, phi, bhspin, model), θo)
    dK_dθo = ForwardDiff.derivative(x -> calculate_kcon(ro, x, phi, i, j, nx, ny, fovx, fovy, bhspin, freq, model), θo)
    XK = MVector{9,Float64}(undef)
    XK[9] = bhspin

    systemODEs_flat = xk -> begin
        Xs = SVector{4}(xk[1], xk[2], xk[3], xk[4])
        Ks = SVector{4}(xk[5], xk[6], xk[7], xk[8])
        spin = xk[9]
        momentum_ode(Xs, Ks, spin, model)
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

"""
    raytrace_gradients_gpu!(d_traj, d_Image, d_dI_dθo, d_dI_dRhigh, i_offset, j_offset,
        block_size_x, block_size_y, ro, θo, phi, bhspin, nx, ny, nmaxstep, freq, fovx,
        fovy, Rout, Rstop, data, params)

GPU kernel launcher for [`calculate_gradients`](@ref): computes the image
intensity together with its derivatives w.r.t. `θo` and `Rhigh`, tiled the
same way as [`Imaging.raytrace_image_gpu!`](@ref).

The calculation is the same as the CPU version ['autodiff_geo_traj_euler_method_grmhd!'](@ref), but on the GPU.
"""
function raytrace_gradients_gpu!(
    d_traj, d_Image, d_dI_dθo, d_dI_dRhigh,
    i_offset, j_offset, block_size_x, block_size_y,
    ro, θo, phi, bhspin, nx, ny, nmaxstep,
    freq, fovx, fovy, Rout, Rstop, data, params
)
    local_i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    local_j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    i = local_i - 1 + i_offset
    j = local_j - 1 + j_offset

    if local_i <= block_size_x && local_j <= block_size_y && i < nx && j < ny
        calculate_gradients(
            d_traj, d_Image, d_dI_dθo, d_dI_dRhigh,
            ro, θo, phi, bhspin, nx, ny, nmaxstep,
            i, j, local_i, local_j, freq, fovx, fovy, Rout, Rstop, params, data
        )
    end
    return nothing
end

"""
    calculate_gradients(traj, d_Image, d_dI_dθo, d_dI_dRhigh, ro, θo, phi, bhspin, nx, ny,
        nmaxstep, i_global, j_global, i_local, j_local, freq, fovx, fovy, Rout, Rstop,
        model, data=nothing)

Per-pixel kernel body for [`raytrace_gradients_gpu!`](@ref). Computes
the per-pixel intensity and its derivatives w.r.t. `θo` and `Rhigh`, by carrying a linearized (`dX`, `dK`)
tangent vector alongside the trajectory (via a single-perturbation
`ForwardDiff.Dual` embedding evaluated through [`momentum_ode`](@ref) each step,
avoiding the need for a materialized Jacobian), and combines it with the
analytic `Rhigh`-derivative from [`Iharm.jar_calc`](@ref) using a
2-component `ForwardDiff.Dual` in the radiative-transfer integration.
"""
function calculate_gradients(
    traj, d_Image, d_dI_dθo, d_dI_dRhigh,
    ro::Float64, θo::Float64, phi::Float64, bhspin::Float64,
    nx::Int64, ny::Int64, nmaxstep::Int64,
    i_global::Int64, j_global::Int64,
    i_local::Int64, j_local::Int64,
    freq::Float64, fovx::Float64, fovy::Float64, Rout::Float64, Rstop::Float64, model,
    data::T_data=nothing
) where {T_data}

    if (i_global >= nx || j_global >= ny)
        return nothing
    end

    dX_dθo = ForwardDiff.derivative(
        x -> Camera.camera_position(ro, x, phi, bhspin, model),
        θo
    )

    dK_dθo = ForwardDiff.derivative(x -> begin
        Xcam_dual = Camera.camera_position(ro, x, phi, bhspin, model)
        K_init = Geodesics.init_kcon(i_global, j_global, Xcam_dual, nx, ny, fovx, fovy, bhspin, model)
        return K_init * (freq * Constants.HPL / (Constants.ME * Constants.CL * Constants.CL))
    end, θo)

    Xcam = SVector{4,Float64}(Camera.camera_position(ro, θo, phi, bhspin, model))
    Kcon = Geodesics.init_kcon(i_global, j_global, Xcam, nx, ny, fovx, fovy, bhspin, model)
    Kcon = Kcon * (freq * Constants.HPL / (Constants.ME * Constants.CL * Constants.CL))

    dl_unit::Float64 = model.L_unit * Constants.HPL / (Constants.ME * Constants.CL^2)
    Rh = 1.0 + sqrt(1.0 - bhspin * bhspin)

    X = Xcam
    K = Kcon
    Xhalf = SVector{4,Float64}(0.0, 0.0, 0.0, 0.0)
    Khalf = SVector{4,Float64}(0.0, 0.0, 0.0, 0.0)

    dX = dX_dθo
    dK = dK_dθo

    step::Int64 = 1
    @inbounds traj[i_local, j_local, step] = GeoTypes.OfTrajGRMHD(
        0.0, X, K, Xhalf, Khalf, dX, dK
    )

    while (Geodesics.stop_backward_integration(X, K, Rh, Rstop) == 0 && (step < nmaxstep))
        @inbounds begin
            dl = Geodesics.stepsize(X, K, model.cstartx, model.cstopx)
            scaled_dl = dl * dl_unit

            X_dual = ForwardDiff.Dual{Nothing}.(X, dX)
            K_dual = ForwardDiff.Dual{Nothing}.(K, dK)

            dK_dual = momentum_ode(X_dual, K_dual, bhspin, model)

            d_dK_dl = SVector{4}(
                ForwardDiff.partials(dK_dual[1], 1),
                ForwardDiff.partials(dK_dual[2], 1),
                ForwardDiff.partials(dK_dual[3], 1),
                ForwardDiff.partials(dK_dual[4], 1)
            )

            next_dX = dX - dl * dK
            next_dK = dK - dl * d_dK_dl

            X, K, Xhalf, Khalf = Geodesics.push_photon(X, K, -dl, bhspin, model)

            dX = next_dX
            dK = next_dK

            step += 1
            traj[i_local, j_local, step] = GeoTypes.OfTrajGRMHD(
                scaled_dl, X, K, Xhalf, Khalf, dX, dK
            )
        end
    end

    Intensity = 0.0
    dI_dθo = 0.0
    dI_dRhigh = 0.0

    @inbounds Xi_S = traj[i_local, j_local, step].X
    @inbounds Kconi_S = traj[i_local, j_local, step].Kcon
    @inbounds dX_dθo_i = traj[i_local, j_local, step].dX_dθo
    @inbounds dK_dθo_i = traj[i_local, j_local, step].dK_dθo

    ji, ki, dji_dRhigh, dki_dRhigh = Radiation.get_jk(Xi_S, Kconi_S, freq, bhspin, model, data, Val(true))

    Xi_dual = ForwardDiff.Dual{Nothing}.(Xi_S, dX_dθo_i)
    Ki_dual = ForwardDiff.Dual{Nothing}.(Kconi_S, dK_dθo_i)
    ji_dual_theta, ki_dual_theta = Iharm.jar_calc_ad(Xi_dual, Ki_dual, bhspin, model, data)

    need_recalc_duals = false

    @inbounds for nstep in step:-1:2
        Xf_S = traj[i_local, j_local, nstep-1].X

        if !Radiation.radiating_region(Xf_S, model, Rh)
            need_recalc_duals = true
            continue
        end

        if need_recalc_duals
            Xi_S_bound = traj[i_local, j_local, nstep].X
            Kconi_S_bound = traj[i_local, j_local, nstep].Kcon
            dX_dθo_i_bound = traj[i_local, j_local, nstep].dX_dθo
            dK_dθo_i_bound = traj[i_local, j_local, nstep].dK_dθo

            Xi_dual_bound = ForwardDiff.Dual{Nothing}.(Xi_S_bound, dX_dθo_i_bound)
            Ki_dual_bound = ForwardDiff.Dual{Nothing}.(Kconi_S_bound, dK_dθo_i_bound)
            ji_dual_theta, ki_dual_theta = Iharm.jar_calc_ad(Xi_dual_bound, Ki_dual_bound, bhspin, model, data)

            need_recalc_duals = false
        end

        Kconf_S = traj[i_local, j_local, nstep-1].Kcon
        dX_dθo_f = traj[i_local, j_local, nstep-1].dX_dθo
        dK_dθo_f = traj[i_local, j_local, nstep-1].dK_dθo
        dl_step = traj[i_local, j_local, nstep].dl

        jf, kf, djf_dRhigh, dkf_dRhigh = Radiation.get_jk(Xf_S, Kconf_S, freq, bhspin, model, data, Val(true))

        Xf_dual = ForwardDiff.Dual{Nothing}.(Xf_S, dX_dθo_f)
        Kf_dual = ForwardDiff.Dual{Nothing}.(Kconf_S, dK_dθo_f)
        jf_dual_theta, kf_dual_theta = Iharm.jar_calc_ad(Xf_dual, Kf_dual, bhspin, model, data)

        P2 = ForwardDiff.Partials{2,Float64}
        ji_comb = ForwardDiff.Dual{Nothing}(ji, P2((ForwardDiff.partials(ji_dual_theta, 1), dji_dRhigh)))
        ki_comb = ForwardDiff.Dual{Nothing}(ki, P2((ForwardDiff.partials(ki_dual_theta, 1), dki_dRhigh)))
        jf_comb = ForwardDiff.Dual{Nothing}(jf, P2((ForwardDiff.partials(jf_dual_theta, 1), djf_dRhigh)))
        kf_comb = ForwardDiff.Dual{Nothing}(kf, P2((ForwardDiff.partials(kf_dual_theta, 1), dkf_dRhigh)))

        I_comb_in = ForwardDiff.Dual{Nothing}(Intensity, P2((dI_dθo, dI_dRhigh)))

        I_comb_out = Radiation.approximate_solve(I_comb_in, ji_comb, ki_comb, jf_comb, kf_comb, dl_step)

        Intensity = ForwardDiff.value(I_comb_out)
        dI_dθo    = ForwardDiff.partials(I_comb_out, 1)
        dI_dRhigh = ForwardDiff.partials(I_comb_out, 2)

        CUDA.@cuassert !(isnan(Intensity) || isinf(Intensity)) "NaN/Inf encountered!"

        ji = jf; ki = kf
        dji_dRhigh = djf_dRhigh; dki_dRhigh = dkf_dRhigh
        ji_dual_theta = jf_dual_theta; ki_dual_theta = kf_dual_theta
    end

    freq3 = freq^3
    @inbounds d_Image[i_global+1, j_global+1]    = Intensity * freq3
    @inbounds d_dI_dθo[i_global+1, j_global+1]   = dI_dθo * freq3
    @inbounds d_dI_dRhigh[i_global+1, j_global+1] = dI_dRhigh * freq3

    return nothing
end


"""
    differentiate_pixel_intensity(θo, i, j, bhspin, model, data, ro, phi, nx, ny,
        fovx, fovy, freq, Rstop, nmaxstep, traj)

Compute pixel `(i, j)`'s intensity and its θo-derivative by seeding `θo` as
a `ForwardDiff.Dual` and calling [`Imaging.calculate_pixel_intensity`](@ref).
`traj` must be a pre-allocated `Vector{GeoTypes.OfTrajDual{ForwardDiff.Dual{Nothing,Float64,1}}}`
sized to `nmaxstep`.

This differs from [`autodiff_geo_traj_euler_method_grmhd!`](@ref) in that it does autodiff through the ODE solver, instead of integrating the sensitivities ODE.
# Returns
- A tuple `(I, dI_dθo)`.
"""
function differentiate_pixel_intensity(θo::Float64, i::Int, j::Int, bhspin, model, data,
    ro, phi, nx::Int, ny::Int, fovx, fovy, freq, Rstop, nmaxstep::Int, traj)
    d = ForwardDiff.Dual{Nothing}(θo, 1.0)
    out = Imaging.calculate_pixel_intensity(traj, ro, d, phi, bhspin, i, j, nx, ny, fovx, fovy, freq, Rstop, nmaxstep, model, data)
    return ForwardDiff.value(out), ForwardDiff.partials(out, 1)
end

end
