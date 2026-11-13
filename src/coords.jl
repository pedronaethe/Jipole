"""
Coordinate transformations between the internal (simulation) coordinates,
Kerr-Schild coordinates, and Boyer-Lindquist coordinates.

The functions that depend on the coordinate mapping used by a given model
(`bl_coord`, `set_ks_jacobian`) are generic on `model::AbstractModel`:
the method defined here is the default (matching the exponential-radial,
uncompressed-polar mapping used by `Analytic`/`ThinDisk`), and `Iharm`
overrides them with its metric-dependent (MKS/FMKS) mapping.

Some of the functions here are reminescent from previous versions of the code and
attempts to integrate with different packages. They are not gonna be deleted because 
they might be useful in the future.
"""
module Coordinates

using StaticArrays
using LinearAlgebra
using ..Constants
using ..AbstractModels

export get_theta_from_x, bl_to_ks, set_ks_jacobian,
      set_ks_jacobian_inverse, vec_from_ks, bl_coord, flip_index

"""
    get_theta_from_x(X, model)

Compute the polar coordinate `th` from the internal coordinates `X`.

# Arguments
- `X`: Position four-vector in internal coordinates.
- `model`: Model parameters, used to select the coordinate mapping.

# Returns
- The polar coordinate `th`.
"""
function get_theta_from_x(X, model)
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
    set_ks_jacobian(X, model)

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
function set_ks_jacobian(X, model::AbstractModel)
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
    set_ks_jacobian_inverse(X, model)

Compute the inverse Jacobian matrix `dX/dx` for the transformation from
internal coordinates to Kerr-Schild coordinates.

# Arguments
- `X`: Position four-vector in internal coordinates.
- `model`: Model parameters, used to select the coordinate mapping.

# Returns
- The inverse Jacobian matrix `dX/dx`.
"""
function set_ks_jacobian_inverse(X, model)
    dxdX = set_ks_jacobian(X, model)
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
    dXdx = set_ks_jacobian_inverse(X, model)

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

end
