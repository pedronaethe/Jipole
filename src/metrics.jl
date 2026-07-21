"""
Metric tensor construction and inversion.

`METRIC_*` are the metric-family identifiers shared by every model's
`metric` field (previously defined redundantly in each model file).
"""
module Metrics

using LinearAlgebra
using StaticArrays
using ..Constants
using ..Coordinates
using ..DebugFunctions

export METRIC_MKS, METRIC_BHACMKS, METRIC_FMKS, METRIC_MKS3, METRIC_EKS,
    METRIC_MINKOWSKI, METRIC_EMINKOWSKI,
    gdet_func, gcov_func!, gcov_func, gcov_func_fd, gcon_func!, gcon_func, gcov_bl!

const METRIC_MKS = 0
const METRIC_BHACMKS = 1
const METRIC_FMKS = 2
const METRIC_MKS3 = 3
const METRIC_EKS = 4
const METRIC_MINKOWSKI = 5
const METRIC_EMINKOWSKI = 6

"""
    gdet_func(gcov)

Compute the square root of the absolute determinant of the covariant
metric tensor, via LU decomposition.

# Arguments
- `gcov`: Covariant metric tensor.

# Returns
- `sqrt(abs(det(gcov)))`, or `-1.0` if `gcov` is numerically singular.
"""
function gdet_func(gcov)
    F = lu(gcov)
    U = F.U

    if any(abs(U[i, i]) < 1e-14 for i in 1:size(U, 1))
        return -1.0
    end

    gdet = prod(diag(U))
    return sqrt(abs(gdet))
end

"""
    gcov_func!(X, bhspin, model, gcov, R0=0.0)

In-place version of [`gcov_func`](@ref), writing the result into `gcov`.

`bhspin` is a separate argument (rather than `model.a`) so this function
stays differentiable when called on the autodiff path.

# Arguments
- `X`: Position four-vector in internal coordinates.
- `bhspin`: Dimensionless black hole spin parameter.
- `model`: Model parameters (metric family via `model.metric`).
- `gcov`: Output matrix, overwritten with the covariant metric tensor.
- `R0`: Optional radial coordinate shift.
"""
function gcov_func!(X, bhspin, model, gcov, R0::Float64=0.0)
    r, th = Coordinates.bl_coord(X, model, R0)
    T = promote_type(typeof(r), typeof(th), typeof(bhspin))
    fill!(gcov, 0.0)

    if model.metric == METRIC_MINKOWSKI
        gcov[1, 1] = -1.0
        gcov[2, 2] = 1.0
        gcov[3, 3] = r * r
        gcov[4, 4] = r * r * sin(th)^2
        return
    elseif model.metric == METRIC_EMINKOWSKI
        gcov[1, 1] = -1.0
        gcov[2, 2] = r * r
        gcov[3, 3] = r * r
        gcov[4, 4] = r * r * sin(th)^2
        return
    end

    Gcov_ks = @MMatrix zeros(T, 4, 4)

    cth = cos(th)
    sth = sin(th)
    s2 = sth^2
    rho2 = r^2 + bhspin^2 * cth^2

    Gcov_ks[1, 1] = -1.0 + 2.0 * r / rho2
    Gcov_ks[1, 2] = 2.0 * r / rho2
    Gcov_ks[1, 4] = -2.0 * bhspin * r * s2 / rho2

    Gcov_ks[2, 1] = Gcov_ks[1, 2]
    Gcov_ks[2, 2] = 1.0 + 2.0 * r / rho2
    Gcov_ks[2, 4] = -bhspin * s2 * (1.0 + 2.0 * r / rho2)

    Gcov_ks[3, 3] = rho2

    Gcov_ks[4, 1] = Gcov_ks[1, 4]
    Gcov_ks[4, 2] = Gcov_ks[2, 4]
    Gcov_ks[4, 4] = s2 * (rho2 + bhspin^2 * s2 * (1.0 + 2.0 * r / rho2))

    dxdX = Coordinates.set_ks_jacobian(X, model)

    fill!(gcov, 0.0)
    for mu in 1:4
        for nu in 1:4
            sum_val = 0.0
            for lam in 1:4
                for kap in 1:4
                    sum_val += Gcov_ks[lam, kap] * dxdX[lam, mu] * dxdX[kap, nu]
                end
            end
            gcov[mu, nu] = sum_val
        end
    end
end

"""
    gcov_func(X, bhspin, model, R0=0.0)

Compute the covariant metric tensor `g_{mu,nu}` at the position `X`.
Adapted from `ipole`'s C code logic.

`bhspin` is a separate argument (rather than `model.a`) so this function
stays differentiable when called on the autodiff path.

# Arguments
- `X`: Position four-vector in internal coordinates.
- `bhspin`: Dimensionless black hole spin parameter.
- `model`: Model parameters (metric family via `model.metric`).
- `R0`: Optional radial coordinate shift.

# Returns
- The covariant metric tensor.
"""
function gcov_func(X, bhspin, model, R0::Float64=0.0)
    r, th = Coordinates.bl_coord(X, model, R0)
    T = promote_type(typeof(r), typeof(th), typeof(bhspin))

    if model.metric == METRIC_MINKOWSKI
        return SMatrix{4,4,T}(
            -1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, r * r, 0.0,
            0.0, 0.0, 0.0, r * r * sin(th)^2
        )
    elseif model.metric == METRIC_EMINKOWSKI
        return SMatrix{4,4,T}(
            -1.0, 0.0, 0.0, 0.0,
            0.0, r * r, 0.0, 0.0,
            0.0, 0.0, r * r, 0.0,
            0.0, 0.0, 0.0, r * r * sin(th)^2
        )
    end

    cth = cos(th)
    sth = sin(th)
    s2 = sth^2
    rho2 = r^2 + bhspin^2 * cth^2

    term1 = 1.0 + 2.0 * r / rho2
    term2 = 2.0 * r / rho2
    term3 = -2.0 * bhspin * r * s2 / rho2
    term4 = -bhspin * s2 * term1

    Gcov_ks = SMatrix{4,4,T}(
        -1.0 + term2, term2, 0.0, term3,
        term2, term1, 0.0, term4,
        0.0, 0.0, rho2, 0.0,
        term3, term4, 0.0, s2 * (rho2 + bhspin^2 * s2 * term1)
    )

    dxdX = Coordinates.set_ks_jacobian(X, model)

    return transpose(dxdX) * Gcov_ks * dxdX
end

"""
    gcov_func_fd(X, bhspin, model, R0=0.0)

Compute the covariant metric tensor via [`Coordinates.gcov_ks`](@ref) and
the Jacobian from [`Coordinates.set_ks_jacobian`](@ref) (used as a finite-
difference-friendly alternative to [`gcov_func`](@ref)).

# Arguments
- `X`: Position four-vector in internal coordinates.
- `bhspin`: Dimensionless black hole spin parameter.
- `model`: Model parameters.
- `R0`: Optional radial coordinate shift.

# Returns
- The covariant metric tensor.
"""
function gcov_func_fd(X, bhspin, model, R0::Float64=0.0)
    r, th = Coordinates.bl_coord(X, model, R0)
    T = promote_type(typeof(r), typeof(th), typeof(bhspin))
    gcov = @MMatrix zeros(T, 4, 4)
    Gcov_ks = Coordinates.gcov_ks(r, th, bhspin)

    dxdX = Coordinates.set_ks_jacobian(X, model)
    for μ in 1:Constants.NDIM
        for ν in 1:Constants.NDIM
            for λ in 1:Constants.NDIM
                for κ in 1:Constants.NDIM
                    gcov[μ, ν] += Gcov_ks[λ, κ] * dxdX[λ, μ] * dxdX[κ, ν]
                end
            end
        end
    end
    return gcov
end

"""
    gcon_func!(gcov, gcon)

In-place version of [`gcon_func`](@ref), writing the result into `gcon`.

# Arguments
- `gcov`: Covariant metric tensor.
- `gcon`: Output matrix, overwritten with the contravariant metric tensor.
"""
function gcon_func!(gcov, gcon)
    gcon .= inv(gcov)
end

"""
    gcon_func(gcov)

Compute the contravariant metric tensor by inverting the covariant metric
tensor.

# Arguments
- `gcov`: Covariant metric tensor.

# Returns
- The contravariant metric tensor.
"""
function gcon_func(gcov)
    return inv(gcov)
end

"""
    gcov_bl!(r, th, bhspin, gcov)

Compute the covariant metric tensor in Boyer-Lindquist coordinates.

# Arguments
- `r`: Radial coordinate in Boyer-Lindquist coordinates.
- `th`: Polar coordinate in Boyer-Lindquist coordinates.
- `bhspin`: Dimensionless black hole spin parameter.
- `gcov`: Output matrix, overwritten with the covariant metric tensor.
"""
function gcov_bl!(r, th, bhspin, gcov)
    sth = sin(th)
    if sth < 1e-40
        sth = 10^(-40)
    end
    cth = cos(th)
    s2 = sth * sth
    if r < 1e-40
        r = 10^(-40)
    end
    a2 = bhspin * bhspin
    r2 = r * r
    DD = (1.0 - 2.0 / r + a2 / r2)
    mu = 1.0 + a2 * cth * cth / r2

    gcov[1, 1] = -(1.0 - 2.0 / (r * mu))
    gcov[1, 4] = -2.0 * bhspin * s2 / (r * mu)
    gcov[4, 1] = gcov[1, 4]
    gcov[2, 2] = mu / (DD)
    gcov[3, 3] = r2 * mu
    gcov[4, 4] = r2 * sth * sth * (1.0 + a2 / r2 + 2.0 * a2 * s2 / (r2 * r * mu))

    if gcov[1, 1] == 0 || gcov[2, 2] == 0 || gcov[3, 3] == 0 || gcov[4, 4] == 0
        @error "Singular gcov encountered in gcov_bl"
        println("sth $sth, cth $cth, r $r, a $bhspin, r2 $r2, a2 $a2, mu $mu, DD $DD")
        DebugFunctions.print_matrix("gcov", gcov)
        error("Singular gcov encountered, cannot compute gcov_bl.")
    end
    if any(isnan.(gcov)) || any(isinf.(gcov))
        @error "Singular gcov encountered in gcov_bl"
        println("sth $sth, cth $cth, r $r, a $bhspin, r2 $r2, a2 $a2, mu $mu, DD $DD")
        DebugFunctions.print_matrix("gcov", gcov)
        error("Singular gcov encountered, cannot compute gcov_bl.")
    end
end

end
