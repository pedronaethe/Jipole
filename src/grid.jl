"""
GRMHD simulation grid indexing and interpolation (used by `Iharm`).
"""
module Grid

using StaticArrays
using LinearAlgebra
using ..Coordinates
using ..Metrics

export Xtoijk_ghost, X_in_domain, ijktoX, interp_scalar, interp_scalar_time, gdet_zone

"""
    Xtoijk_ghost(X, model)

Locate the grid zone and interpolation weights for the position `X` on
`model`'s simulation grid.

# Arguments
- `X`: Position four-vector in internal coordinates.
- `model`: Model parameters, providing the grid geometry
  (`model.startx`, `model.dx`, `model.N1`, `model.N2`, `model.N3`,
  `model.cstopx`).

# Returns
- A tuple `(i, j, k, del2, del3, del4)`: the 1-based zone indices and the
  interpolation weights within that zone.
"""
function Xtoijk_ghost(X, model)
    i_logical = trunc(Int, ((X[2] - model.startx[2]) / model.dx[2]) - 0.5 + 1000) - 1000
    j_logical = trunc(Int, ((X[3] - model.startx[3]) / model.dx[3]) - 0.5 + 1000) - 1000

    i = clamp(i_logical, 0, model.N1 - 2)
    j = clamp(j_logical, 0, model.N2 - 2)

    del2 = clamp((X[2] - ((i + 0.5) * model.dx[2] + model.startx[2])) / model.dx[2], 0.0, 1.0)
    del3 = clamp((X[3] - ((j + 0.5) * model.dx[3] + model.startx[3])) / model.dx[3], 0.0, 1.0)

    i += 1
    j += 1

    phi = rem(X[4], model.cstopx[4])
    if phi < 0.0
        phi += model.cstopx[4]
    end

    k_logical = trunc(Int, ((phi - model.startx[4]) / model.dx[4]) - 0.5 + 1000) - 1000
    del4 = clamp((phi - ((k_logical + 0.5) * model.dx[4] + model.startx[4])) / model.dx[4], 0.0, 1.0)

    k_logical_wrapped = mod(k_logical, model.N3)
    k = k_logical_wrapped + 1

    return i, j, k, del2, del3, del4
end

"""
    X_in_domain(X, model)

Check whether the position `X` lies within `model`'s simulation grid
bounds.

# Arguments
- `X`: Position four-vector in internal coordinates.
- `model`: Model parameters, providing the grid bounds (`model.cstartx`,
  `model.cstopx`).

# Returns
- `1` if `X` is within bounds, `0` otherwise.
"""
function X_in_domain(X, model)
    if X[2] < model.cstartx[2] || X[2] > model.cstopx[2] || X[3] < model.cstartx[3] || X[3] > model.cstopx[3]
        return 0
    end
    return 1
end

"""
    ijktoX(i, j, k, X, model)

Compute the internal coordinates `X` at the center of grid zone
`(i, j, k)`.

# Arguments
- `i`, `j`, `k`: 0-based grid zone indices.
- `X`: Output vector, overwritten with the internal coordinates.
- `model`: Model parameters, providing the grid geometry (`model.startx`,
  `model.dx`).
"""
function ijktoX(i, j, k, X, model)
    X[2] = model.startx[2] + (i + 0.5) * model.dx[2]
    X[3] = model.startx[3] + (j + 0.5) * model.dx[3]
    X[4] = model.startx[4] + (k + 0.5) * model.dx[4]
    return
end

"""
    interp_scalar(X, data, model)

Quadrilinearly interpolate the scalar field `data` at the position `X`.

# Arguments
- `X`: Position four-vector in internal coordinates.
- `data`: Scalar field defined on `model`'s simulation grid.
- `model`: Model parameters, providing the grid geometry.

# Returns
- The interpolated scalar value.
"""
function interp_scalar(X, data, model)
    i, j, k, del2, del3, del4 = Xtoijk_ghost(X, model)

    (N1_data, N2_data, N3_data) = size(data)

    ip1 = i + 1
    jp1 = j + 1

    kp1 = k + 1
    if kp1 > N3_data
        kp1 = 1
    end

    b1 = 1.0 - del2
    b2 = 1.0 - del3

    interp = data[i, j, k] * b1 * b2 +
             data[ip1, j, k] * del2 * b2 +
             data[i, jp1, k] * b1 * del3 +
             data[ip1, jp1, k] * del2 * del3

    interp = interp * (1.0 - del4) + (
                 data[i, j, kp1] * b1 * b2 +
                 data[ip1, j, kp1] * del2 * b2 +
                 data[i, jp1, kp1] * b1 * del3 +
                 data[ip1, jp1, kp1] * del2 * del3
             ) * del4

    return interp
end

"""
    interp_scalar_time(X, dataA, dataB, tfac, slow_light, model)

Interpolate the scalar field spatially at `X`, and (if `slow_light` is
set) linearly in time between the `dataA` and `dataB` snapshots.

# Arguments
- `X`: Position four-vector in internal coordinates.
- `dataA`: Scalar field at the earlier time snapshot.
- `dataB`: Scalar field at the later time snapshot.
- `tfac`: Time interpolation factor.
- `slow_light`: Whether slow-light (time-dependent) interpolation is
  enabled.
- `model`: Model parameters, providing the grid geometry.

# Returns
- The interpolated scalar value.
"""
function interp_scalar_time(X, dataA, dataB, tfac, slow_light::Bool, model)
    vA = interp_scalar(X, dataA, model)
    if slow_light
        vB = interp_scalar(X, dataB, model)
        return (tfac) * vA + (1.0 - tfac) * vB
    end
    return vA
end

"""
    gdet_zone(i, j, k, model)

Compute `sqrt(abs(det(gcov)))` at the center of grid zone `(i, j, k)`.

# Arguments
- `i`, `j`, `k`: 0-based grid zone indices.
- `model`: Model parameters, providing the grid geometry and spin.

# Returns
- `sqrt(abs(det(gcov)))` at the zone center.
"""
function gdet_zone(i, j, k, model)
    X2 = model.startx[2] + (i + 0.5) * model.dx[2]
    X3 = model.startx[3] + (j + 0.5) * model.dx[3]
    X4 = model.startx[4] + (k + 0.5) * model.dx[4]

    X = SVector(0.0, X2, X3, X4)

    rt = zeros(MVector{2,Float64})
    Coordinates.bl_coord!(rt, X, model)
    r = rt[1]
    th = rt[2]

    gcovKS = Coordinates.gcov_ks(r, th, model.a)

    dxdX = Coordinates.set_dxdX(X, model)

    gcov = transpose(dxdX) * gcovKS * dxdX
    return Metrics.gdet_func(gcov)
end

end
