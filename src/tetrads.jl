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

export make_camera_tetrad, make_plasma_tetrad, null_normalize, tetrad_to_coordinate!

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

    trial = zero(MVector{4,Float64})
    trial[1] = -1.0

    Ucam = Coordinates.flip_index(trial, Gcon)

    trial = zero(MVector{4,Float64})
    trial[1] = 1.0
    trial[2] = 1.0

    Kcon = Coordinates.flip_index(trial, Gcon)
    trial = zero(MVector{4,Float64})
    trial[3] = 1.0
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
    Econ = MMatrix{4,4,eltype(Ucon)}(undef)
    Ecov = MMatrix{4,4,eltype(Ucon)}(undef)
    ones_vector = ones(MVector{4,Float64})
    Econ[1, :] = Utils.set_Econ_from_trial(1, Ucon)
    Econ[2, :] = Utils.set_Econ_from_trial(4, ones_vector)
    Econ[3, :] = Utils.set_Econ_from_trial(3, Bcon)
    Econ[4, :] = Utils.set_Econ_from_trial(4, Kcon)
    Econ[1, :] = Utils.normalize_vector(Econ[1, :], Gcov)
    Econ[4, :] = Utils.project_out(Econ[4, :], Econ[1, :], Gcov)
    Econ[4, :] = Utils.project_out(Econ[4, :], Econ[1, :], Gcov)
    Econ[4, :] = Utils.normalize_vector(Econ[4, :], Gcov)

    Econ[3, :] = Utils.project_out(Econ[3, :], Econ[1, :], Gcov)
    Econ[3, :] = Utils.project_out(Econ[3, :], Econ[4, :], Gcov)
    Econ[3, :] = Utils.project_out(Econ[3, :], Econ[1, :], Gcov)
    Econ[3, :] = Utils.project_out(Econ[3, :], Econ[4, :], Gcov)
    Econ[3, :] = Utils.normalize_vector(Econ[3, :], Gcov)
    Econ[2, :] = Utils.project_out(Econ[2, :], Econ[1, :], Gcov)
    Econ[2, :] = Utils.project_out(Econ[2, :], Econ[3, :], Gcov)
    Econ[2, :] = Utils.project_out(Econ[2, :], Econ[4, :], Gcov)
    Econ[2, :] = Utils.project_out(Econ[2, :], Econ[1, :], Gcov)
    Econ[2, :] = Utils.project_out(Econ[2, :], Econ[3, :], Gcov)
    Econ[2, :] = Utils.project_out(Econ[2, :], Econ[4, :], Gcov)
    Econ[2, :] = Utils.normalize_vector(Econ[2, :], Gcov)
    oddflag::Int = 0
    flag, dot_var = Utils.check_handedness(Econ, Gcov)

    if flag != 0
        oddflag |= 0x10
    end

    if abs(abs(dot_var) - 1) > 1e-10
        oddflag |= 0x1
    end

    if dot_var < 0
        for k in 1:4
            Econ[2, k] *= -1
        end
    end

    for k in 1:4
        Ecov[k, :] = Coordinates.flip_index(Econ[k, :], Gcov)
    end

    for l in 1:4
        Ecov[1, l] *= -1
    end

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
    Kcon_out = copy(Kcon)
    inorm = sqrt(sum(Kcon[2:4] .^ 2))
    Kcon_out[1] = fnorm
    for k in 2:4
        Kcon_out[k] *= fnorm / inorm
    end
    return Kcon_out
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

end
