"""
Construction of orthonormal tetrads used to define the observer/plasma
rest frames.
"""
module Tetrads

using StaticArrays
using ..Constants
using ..Coordinates
using ..Metrics
using ..Utils

export make_camera_tetrad, make_plasma_tetrad, null_normalize, tetrad_to_coordinate!, tetrad_to_coordinate

"""
    make_camera_tetrad(X, bhspin, model)

Compute the camera tetrad at the position `X`.

The tetrad is constructed such that `e^0` is aligned with the camera's
`Ucam`, `e^3` is aligned with the outward radial direction, `e^2` is
aligned with the north pole of the coordinate system ("y" in the image
plane), and `e^1` is the remaining direction ("x" in the image plane).
Points the camera such that the angular momentum at the field-of-view
center is zero.

# Arguments
- `X`: Position four-vector in internal coordinates.
- `bhspin`: Dimensionless black hole spin parameter.
- `model`: Model parameters, used to select the coordinate mapping.

# Returns
- A tuple `(oddflag, Econ, Ecov)`.
"""
function make_camera_tetrad(X, bhspin, model)
    Gcov = Metrics.gcov_func(X, bhspin, model)
    Gcon = Metrics.gcon_func(Gcov)

    trial = SVector{4,Float64}(-1.0, 0.0, 0.0, 0.0)

    Ucam = Coordinates.flip_index(trial, Gcon)

    trial = SVector{4,Float64}(1.0, 1.0, 0.0, 0.0)

    Kcon = Coordinates.flip_index(trial, Gcon)
    trial = SVector{4,Float64}(0.0, 0.0, 1.0, 0.0)
    sing, Econ, Ecov = make_plasma_tetrad(Ucam, Kcon, trial, Gcov)

    return sing, Econ, Ecov
end

"""
    make_plasma_tetrad(Ucon, Kcon, Bcon, Gcov)

Compute the plasma tetrad from the fluid 4-velocity, photon 4-momentum,
and magnetic field 4-vector.

`Econ[k, l]`: `k` is the tetrad basis index, `l` is the coordinate basis
index (up). `Ecov` switches both indices. `e^0` is along `U`, `e^2` is
along `b`, `e^3` is along the spatial part of `K`.

# Arguments
- `Ucon`: Contravariant 4-velocity of the plasma.
- `Kcon`: Contravariant 4-vector in the direction of the photon.
- `Bcon`: Contravariant 4-vector in the direction of the magnetic field.
- `Gcov`: Covariant metric tensor.

# Returns
- A tuple `(oddflag, Econ, Ecov)`.
"""
function make_plasma_tetrad(Ucon, Kcon, Bcon, Gcov)
    T = eltype(Ucon)
    ones_vector = SVector{4,T}(1.0, 1.0, 1.0, 1.0)

    e1 = Utils.set_Econ_from_trial(1, Ucon)
    e2 = Utils.set_Econ_from_trial(4, ones_vector)
    e3 = Utils.set_Econ_from_trial(3, Bcon)
    e4 = Utils.set_Econ_from_trial(4, Kcon)

    e1 = Utils.normalize_vector(e1, Gcov)

    e4 = Utils.project_out(e4, e1, Gcov)
    e4 = Utils.project_out(e4, e1, Gcov)
    e4 = Utils.normalize_vector(e4, Gcov)

    e3 = Utils.project_out(e3, e1, Gcov)
    e3 = Utils.project_out(e3, e4, Gcov)
    e3 = Utils.project_out(e3, e1, Gcov)
    e3 = Utils.project_out(e3, e4, Gcov)
    e3 = Utils.normalize_vector(e3, Gcov)

    e2 = Utils.project_out(e2, e1, Gcov)
    e2 = Utils.project_out(e2, e3, Gcov)
    e2 = Utils.project_out(e2, e4, Gcov)
    e2 = Utils.project_out(e2, e1, Gcov)
    e2 = Utils.project_out(e2, e3, Gcov)
    e2 = Utils.project_out(e2, e4, Gcov)
    e2 = Utils.normalize_vector(e2, Gcov)

    Econ_tmp = SMatrix{4,4,T}(
        e1[1], e2[1], e3[1], e4[1],
        e1[2], e2[2], e3[2], e4[2],
        e1[3], e2[3], e3[3], e4[3],
        e1[4], e2[4], e3[4], e4[4]
    )

    oddflag::Int = 0
    flag, dot_var = Utils.check_handedness(Econ_tmp, Gcov)

    if flag != 0
        oddflag |= 0x10
    end

    if abs(abs(dot_var) - 1) > 1e-10
        oddflag |= 0x1
    end

    if dot_var < 0
        e2 = -e2
    end

    ecov1 = Coordinates.flip_index(e1, Gcov)
    ecov2 = Coordinates.flip_index(e2, Gcov)
    ecov3 = Coordinates.flip_index(e3, Gcov)
    ecov4 = Coordinates.flip_index(e4, Gcov)

    ecov1 = -ecov1

    Econ = SMatrix{4,4,T}(
        e1[1], e2[1], e3[1], e4[1],
        e1[2], e2[2], e3[2], e4[2],
        e1[3], e2[3], e3[3], e4[3],
        e1[4], e2[4], e3[4], e4[4]
    )

    Ecov = SMatrix{4,4,T}(
        ecov1[1], ecov2[1], ecov3[1], ecov4[1],
        ecov1[2], ecov2[2], ecov3[2], ecov4[2],
        ecov1[3], ecov2[3], ecov3[3], ecov4[3],
        ecov1[4], ecov2[4], ecov3[4], ecov4[4]
    )

    return oddflag, Econ, Ecov
end

"""
    null_normalize(Kcon, fnorm)

Normalize a null vector in a tetrad frame.

# Arguments
- `Kcon`: 4-vector in the tetrad frame.
- `fnorm`: Desired norm of the vector.

# Returns
- The normalized vector.
"""
function null_normalize(Kcon, fnorm)
    T = promote_type(eltype(Kcon), typeof(fnorm))
    inorm = sqrt(Kcon[2]^2 + Kcon[3]^2 + Kcon[4]^2)
    scale = fnorm / inorm
    return SVector{4,T}(fnorm, Kcon[2] * scale, Kcon[3] * scale, Kcon[4] * scale)
end

"""
    tetrad_to_coordinate!(Kcon, Econ, Kcon_tetrad)

Convert a contravariant 4-vector from the tetrad frame to the coordinate
frame.

# Arguments
- `Kcon`: Output vector, overwritten with the coordinate-frame 4-vector.
- `Econ`: Tetrad basis vectors in covariant form.
- `Kcon_tetrad`: Contravariant 4-vector in the tetrad frame.

# Returns
- `Kcon`.
"""
function tetrad_to_coordinate!(Kcon, Econ, Kcon_tetrad)
    @inbounds for l in 1:4
        Kcon[l] = Econ[1, l] * Kcon_tetrad[1] + Econ[2, l] * Kcon_tetrad[2] + Econ[3, l] * Kcon_tetrad[3] + Econ[4, l] * Kcon_tetrad[4]
    end
    return Kcon
end

"""
    tetrad_to_coordinate(Econ, Kcon_tetrad)

Non-mutating version of [`tetrad_to_coordinate!`](@ref), returning the
coordinate-frame 4-vector directly instead of writing into a caller-
supplied buffer.

# Arguments
- `Econ`: Tetrad basis vectors in covariant form.
- `Kcon_tetrad`: Contravariant 4-vector in the tetrad frame.

# Returns
- The coordinate-frame 4-vector.
"""
function tetrad_to_coordinate(Econ, Kcon_tetrad)
    return SVector{4,eltype(Econ)}(
        Econ[1, 1] * Kcon_tetrad[1] + Econ[2, 1] * Kcon_tetrad[2] + Econ[3, 1] * Kcon_tetrad[3] + Econ[4, 1] * Kcon_tetrad[4],
        Econ[1, 2] * Kcon_tetrad[1] + Econ[2, 2] * Kcon_tetrad[2] + Econ[3, 2] * Kcon_tetrad[3] + Econ[4, 2] * Kcon_tetrad[4],
        Econ[1, 3] * Kcon_tetrad[1] + Econ[2, 3] * Kcon_tetrad[2] + Econ[3, 3] * Kcon_tetrad[3] + Econ[4, 3] * Kcon_tetrad[4],
        Econ[1, 4] * Kcon_tetrad[1] + Econ[2, 4] * Kcon_tetrad[2] + Econ[3, 4] * Kcon_tetrad[3] + Econ[4, 4] * Kcon_tetrad[4]
    )
end

end
