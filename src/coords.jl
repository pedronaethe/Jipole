"""
Coordinate transformations between the internal (simulation) coordinates,
Kerr-Schild coordinates, and Boyer-Lindquist coordinates.

The functions that depend on the coordinate mapping used by a given model
(`bl_coord`, `bl_coord!`, `set_dxdX`) are generic on `model::AbstractModel`:
the method defined here is the default (matching the exponential-radial,
uncompressed-polar mapping used by `Analytic`/`ThinDisk`), and `Iharm`
overrides them with its metric-dependent (MKS/FMKS) mapping.
"""
module Coordinates

using StaticArrays
using LinearAlgebra
using ..Constants
using ..AbstractModels

export get_theta_from_X, bl_to_ks, ks_to_bl, vec_to_bl, vec_to_ks, set_dxdX,
    gcov_ks, set_dXdx, vec_from_ks, bl_coord, bl_coord!, flip_index, flip_index!

"""
    get_theta_from_X(X, model)

Compute the polar coordinate `th` from the internal coordinates `X`.

# Arguments
- `X`: Position four-vector in internal coordinates.
- `model`: Model parameters, used to select the coordinate mapping.

# Returns
- The polar coordinate `th`.
"""
function get_theta_from_X(X, model)
    _, th = bl_coord(X, model)
    return th
end

"""
    bl_to_ks(X, ucon_bl, bhspin, model)

Convert a contravariant 4-vector from Boyer-Lindquist to Kerr-Schild
coordinates.

`bhspin` is a separate argument (rather than `model.a`) so this function
stays differentiable when called on the autodiff path.

# Arguments
- `X`: Position four-vector in internal coordinates.
- `ucon_bl`: Contravariant 4-vector in Boyer-Lindquist coordinates.
- `bhspin`: Dimensionless black hole spin parameter.
- `model`: Model parameters, used to select the coordinate mapping.

# Returns
- The contravariant 4-vector in Kerr-Schild coordinates.
"""
function bl_to_ks(X, ucon_bl, bhspin, model)
    T = promote_type(eltype(X), eltype(ucon_bl), typeof(bhspin))
    ucon_ks = MVector{4,T}((zero(T), zero(T), zero(T), zero(T)))

    r, _ = bl_coord(X, model)
    trans = MMatrix{4,4,T}(undef)
    for μ in 1:4
        for ν in 1:4
            trans[μ, ν] = μ == ν ? one(T) : zero(T)
        end
    end

    denom = r * r - 2.0 * r + bhspin * bhspin
    trans[1, 2] = 2.0 * r / denom
    trans[4, 2] = bhspin / denom

    for μ in 1:4
        for ν in 1:4
            ucon_ks[μ] += trans[μ, ν] * ucon_bl[ν]
        end
    end

    return ucon_ks
end

"""
    ks_to_bl(X, ucon_ks, bhspin, model)

Convert a contravariant 4-vector from Kerr-Schild to Boyer-Lindquist
coordinates.

`bhspin` is a separate argument (rather than `model.a`) so this function
stays differentiable when called on the autodiff path.

# Arguments
- `X`: Position four-vector in internal coordinates.
- `ucon_ks`: Contravariant 4-vector in Kerr-Schild coordinates.
- `bhspin`: Dimensionless black hole spin parameter.
- `model`: Model parameters, used to select the coordinate mapping.

# Returns
- The contravariant 4-vector in Boyer-Lindquist coordinates.
"""
function ks_to_bl(X, ucon_ks, bhspin, model)
    ucon_bl = zero(MVector{4,Float64})
    r, _ = bl_coord(X, model)

    trans = MMatrix{4,4,Float64}(undef)
    for μ in 1:Constants.NDIM
        for ν in 1:Constants.NDIM
            trans[μ, ν] = μ == ν ? 1.0 : 0.0
        end
    end

    trans[1, 2] = 2.0 * r / (r * r - 2.0 * r + bhspin * bhspin)
    trans[4, 2] = bhspin / (r * r - 2.0 * r + bhspin * bhspin)

    rev_trans = inv(trans)

    for μ in 1:Constants.NDIM
        ucon_bl[μ] = 0.0
        for ν in 1:Constants.NDIM
            ucon_bl[μ] += rev_trans[μ, ν] * ucon_ks[ν]
        end
    end

    return ucon_bl
end

"""
    vec_to_bl(X, v_nat, bhspin, model)

Convert a 4-vector from the native (internal) coordinate system to
Boyer-Lindquist coordinates, via Kerr-Schild coordinates.

`bhspin` is a separate argument (rather than `model.a`) so this function
stays differentiable when called on the autodiff path.

# Arguments
- `X`: Position four-vector in internal coordinates.
- `v_nat`: 4-vector in the native coordinate system.
- `bhspin`: Dimensionless black hole spin parameter.
- `model`: Model parameters, used to select the coordinate mapping.

# Returns
- The 4-vector in Boyer-Lindquist coordinates.
"""
function vec_to_bl(X, v_nat, bhspin, model)
    v_ks = zero(MVector{4,Float64})
    dxdX = set_dxdX(X, model)
    for μ in 1:Constants.NDIM
        for ν in 1:Constants.NDIM
            v_ks[μ] += dxdX[μ, ν] * v_nat[ν]
        end
    end

    vec_bl = zero(MVector{4,Float64})
    r, _ = bl_coord(X, model)
    trans = MMatrix{4,4,Float64}(undef)
    for μ in 1:Constants.NDIM
        for ν in 1:Constants.NDIM
            trans[μ, ν] = μ == ν ? 1.0 : 0.0
        end
    end
    trans[1, 2] = 2.0 * r / (r * r - 2.0 * r + bhspin * bhspin)
    trans[4, 2] = bhspin / (r * r - 2.0 * r + bhspin * bhspin)

    rev_trans = inv(trans)

    for μ in 1:Constants.NDIM
        vec_bl[μ] = 0.0
        for ν in 1:Constants.NDIM
            vec_bl[μ] += rev_trans[μ, ν] * v_ks[ν]
        end
    end

    return vec_bl
end

"""
    vec_to_ks(X, v_nat, model)

Convert a 4-vector from the native (internal) coordinate system to
Kerr-Schild coordinates.

# Arguments
- `X`: Position four-vector in internal coordinates.
- `v_nat`: 4-vector in the native coordinate system.
- `model`: Model parameters, used to select the coordinate mapping.

# Returns
- The 4-vector in Kerr-Schild coordinates.
"""
function vec_to_ks(X, v_nat, model)
    v_ks = zero(MVector{4,Float64})
    dxdX = set_dxdX(X, model)

    for μ in 1:Constants.NDIM
        for ν in 1:Constants.NDIM
            v_ks[μ] += dxdX[μ, ν] * v_nat[ν]
        end
    end

    return v_ks
end

"""
    set_dxdX(X, model)

Compute the Jacobian matrix `dx/dX` for the transformation from Kerr-Schild
coordinates to internal coordinates. This default method implements the
uncompressed exponential-radial mapping used by `Analytic`/`ThinDisk`;
`Iharm` provides a metric-dependent override.

# Arguments
- `X`: Position four-vector in internal coordinates.
- `model`: Model parameters, used to select the coordinate mapping.

# Returns
- The Jacobian matrix `dx/dX` as an `SMatrix`.
"""
function set_dxdX(X, model::AbstractModel)
    T = eltype(X)

    m22 = exp(X[2])
    m33 = T(π)

    if m33 <= 0.0
        m33 = T(1.0e-10)
    end

    return SMatrix{4,4,T}(
        one(T), zero(T), zero(T), zero(T),
        zero(T), m22, zero(T), zero(T),
        zero(T), zero(T), m33, zero(T),
        zero(T), zero(T), zero(T), one(T)
    )
end

"""
    gcov_ks(r, th, bhspin)

Compute the covariant Kerr-Schild metric tensor at Boyer-Lindquist radius
`r` and polar angle `th`.

# Arguments
- `r`: Radial coordinate in Boyer-Lindquist coordinates.
- `th`: Polar coordinate in Boyer-Lindquist coordinates.
- `bhspin`: Dimensionless black hole spin parameter.

# Returns
- The covariant metric tensor as an `SMatrix`.
"""
Base.@inline function gcov_ks(r, th, bhspin)
    gcov = @MMatrix zeros(Float64, 4, 4)
    cth = cos(th)
    sth = sin(th)

    s2 = sth * sth
    rho2 = r * r + bhspin * bhspin * cth * cth

    gcov[1, 1] = -1.0 + 2.0 * r / rho2
    gcov[1, 2] = 2.0 * r / rho2
    gcov[1, 4] = -2.0 * bhspin * r * s2 / rho2

    gcov[2, 1] = gcov[1, 2]
    gcov[2, 2] = 1.0 + 2.0 * r / rho2
    gcov[2, 4] = -bhspin * s2 * (1.0 + 2.0 * r / rho2)

    gcov[3, 3] = rho2

    gcov[4, 1] = gcov[1, 4]
    gcov[4, 2] = gcov[2, 4]
    gcov[4, 4] = s2 * (rho2 + bhspin * bhspin * s2 * (1.0 + 2.0 * r / rho2))
    return SMatrix(gcov)
end

"""
    set_dXdx(X, model)

Compute the inverse Jacobian matrix `dX/dx` for the transformation from
internal coordinates to Kerr-Schild coordinates.

# Arguments
- `X`: Position four-vector in internal coordinates.
- `model`: Model parameters, used to select the coordinate mapping.

# Returns
- The inverse Jacobian matrix `dX/dx`.
"""
function set_dXdx(X, model)
    dxdX = set_dxdX(X, model)
    return inv(dxdX)
end

"""
    vec_from_ks(X, v_ks, model)

Convert a 4-vector from Kerr-Schild coordinates to the native (internal)
coordinate system.

# Arguments
- `X`: Position four-vector in internal coordinates.
- `v_ks`: 4-vector in Kerr-Schild coordinates.
- `model`: Model parameters, used to select the coordinate mapping.

# Returns
- The 4-vector in the native coordinate system.
"""
function vec_from_ks(X, v_ks, model)
    v_nat = zero(MVector{4,eltype(v_ks)})
    dXdx = set_dXdx(X, model)

    for μ in 1:Constants.NDIM
        for ν in 1:Constants.NDIM
            v_nat[μ] += dXdx[μ, ν] * v_ks[ν]
        end
    end

    return v_nat
end

"""
    bl_coord(X, model, R0=0.0)

Compute the Boyer-Lindquist coordinates `(r, th)` from the internal
coordinates `X`. This default method implements the uncompressed
exponential-radial mapping used by `Analytic`/`ThinDisk`; `Iharm` provides
a metric-dependent override.

# Arguments
- `X`: Position four-vector in internal coordinates.
- `model`: Model parameters, used to select the coordinate mapping.
- `R0`: Optional radial coordinate shift.

# Returns
- A tuple `(r, th)`.
"""
function bl_coord(X, model::AbstractModel, R0::Float64=0.0)
    r = exp(X[2]) + R0
    th = π * X[3]
    return r, th
end

"""
    bl_coord!(rt, X, model, R0=0.0)

In-place version of [`bl_coord`](@ref), writing `(r, th)` into `rt`.

# Arguments
- `rt`: Output vector, overwritten with `(r, th)`.
- `X`: Position four-vector in internal coordinates.
- `model`: Model parameters, used to select the coordinate mapping.
- `R0`: Optional radial coordinate shift.
"""
Base.@inline function bl_coord!(rt, X, model::AbstractModel, R0::Float64=0.0)
    rt[1] = exp(X[2]) + R0
    rt[2] = π * X[3]
end

"""
    flip_index(vector, metric)

Lower (or raise) the index of a 4-vector using the given metric tensor.

# Arguments
- `vector`: 4-vector whose index is to be flipped.
- `metric`: Metric tensor used for flipping.

# Returns
- The flipped 4-vector.
"""
function flip_index(vector, metric)
    return metric * vector
end

"""
    flip_index!(flipped_vector, vector, metric)

In-place version of [`flip_index`](@ref).

# Arguments
- `flipped_vector`: Output vector, overwritten with the flipped result.
- `vector`: 4-vector whose index is to be flipped.
- `metric`: Metric tensor used for flipping.
"""
@inline function flip_index!(flipped_vector, vector, metric)
    @inbounds for ν in 1:Constants.NDIM
        s = zero(eltype(vector))
        for μ in 1:Constants.NDIM
            s += metric[ν, μ] * vector[μ]
        end
        flipped_vector[ν] = s
    end
end

end
