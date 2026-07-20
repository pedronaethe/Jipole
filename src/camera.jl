"""
Camera placement in native simulation coordinates.
"""
module Camera

using StaticArrays
using LinearAlgebra
using ..Constants
using ..Coordinates
using ..AbstractModels

export camera_position, root_find_std, root_find_ipole

"""
    root_find_std(x, model)

Compute the native polar coordinate corresponding to a target physical
polar angle.

The solution is first estimated using [`root_find_ipole`](@ref) and then
refined with a single Newton iteration, where the derivative of
[`Coordinates.get_theta_from_X`](@ref) is approximated using finite
differences.

# Arguments
- `x`: Position four-vector in physical coordinates.
- `model`: Model parameters, providing the native grid bounds
  (`model.cstartx`, `model.cstopx`).

# Returns
- Native polar coordinate corresponding to the requested physical
  polar angle.
"""
function root_find_std(x, model)
    x3_val = root_find_ipole(x, model)

    T = eltype(x)
    c1 = zero(T)
    c2 = log(x[2])
    c4 = x[4]

    x_eval = @SVector [c1, c2, x3_val, c4]

    eps = 1e-7

    x_eval_eps = @SVector [c1, c2, x3_val + eps, c4]

    slope = (Coordinates.get_theta_from_X(x_eval_eps, model) - Coordinates.get_theta_from_X(x_eval, model)) / eps

    return x3_val - (Coordinates.get_theta_from_X(x_eval, model) - x[3]) / slope
end

"""
    root_find_ipole(x, model)

Compute the native polar coordinate corresponding to a target physical
polar angle using the bisection method.

The search is performed over the interval defined by `model.cstartx` and
`model.cstopx` until the requested tolerance is reached. This routine
provides the initial estimate used by [`root_find_std`](@ref).

# Arguments
- `x`: Position four-vector containing the target physical polar angle.
- `model`: Model parameters, providing the native grid bounds
  (`model.cstartx`, `model.cstopx`).

# Returns
- Native polar coordinate corresponding to the requested physical
  polar angle.
"""
function root_find_ipole(x, model)
    th = x[3]
    T = eltype(x)
    cstartx = model.cstartx
    cstopx = model.cstopx

    c1 = zero(T)
    c2 = log(x[2])
    c4 = x[4]

    xa3 = zero(T)
    xb3 = zero(T)

    if x[3] < π / 2.0
        xa3 = cstartx[3]
        xb3 = (cstopx[3] - cstartx[3]) / 2 + Constants.SMALL
    else
        xa3 = (cstopx[3] - cstartx[3]) / 2 - Constants.SMALL
        xb3 = cstopx[3]
    end

    tol::Float64 = 1.e-9

    tha = Coordinates.get_theta_from_X(@SVector[c1, c2, xa3, c4], model)
    thb = Coordinates.get_theta_from_X(@SVector[c1, c2, xb3, c4], model)

    if abs(tha - th) < tol
        return xa3
    elseif abs(thb - th) < tol
        return xb3
    end

    xc3 = zero(T)

    for i in 1:1000
        xc3 = 0.5 * (xa3 + xb3)
        thc = Coordinates.get_theta_from_X(@SVector[c1, c2, xc3, c4], model)

        if (thc - th) * (thb - th) < 0.0
            xa3 = xc3
        else
            xb3 = xc3
        end

        err = thc - th
        if abs(err) < tol
            break
        end
    end

    return xc3
end

"""
    camera_position(cam_dist, cam_theta_angle, cam_phi_angle, bhspin, model)

Compute the camera position in the native simulation coordinates.

This default method (used by `Analytic`/`ThinDisk`) obtains the camera
coordinates directly from the input viewing geometry. `Iharm` overrides
this with a numerical inversion of the coordinate mapping, via
[`root_find_std`](@ref).

`bhspin` is kept as an explicit argument (rather than reading `model.a`)
so that it can carry a `ForwardDiff.Dual` when this function sits on the
autodiff path (see `Autodiff.AutoDiffGeoTrajEulerMethod!`).

# Arguments
- `cam_dist`: Radial distance of the camera from the black hole.
- `cam_theta_angle`: Polar viewing angle in degrees.
- `cam_phi_angle`: Azimuthal viewing angle in degrees.
- `bhspin`: Dimensionless black hole spin parameter.
- `model`: Model parameters, used to select the coordinate mapping.

# Returns
- `SVector{4}` containing the camera position in native simulation
  coordinates.
"""
function camera_position(cam_dist, cam_theta_angle, cam_phi_angle, bhspin, model::AbstractModel)
    T = promote_type(typeof(cam_dist), typeof(cam_theta_angle), typeof(cam_phi_angle), typeof(bhspin))
    return @SVector [zero(T), log(cam_dist), cam_theta_angle / 180, (cam_phi_angle / 180) * π]
end

end
