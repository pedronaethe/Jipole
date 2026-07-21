"""
Small linear-algebra helpers used when building tetrads.
"""
module Utils

using StaticArrays
using LinearAlgebra
using ..Constants

export set_Econ_from_trial, normalize_vector, project_out, levi_civita, check_handedness

"""
    set_Econ_from_trial(defdir, trial)

Copy the trial vector; if its spatial norm is too small, fall back to the
unit vector along `defdir`.

# Arguments
- `defdir`: Direction to default to if `trial` is nearly null.
- `trial`: Trial vector to set the tetrad vector from.

# Returns
- The resulting tetrad basis vector.
"""
function set_Econ_from_trial(defdir::Int, trial)
    T = eltype(trial)
    norm = abs(trial[2]) + abs(trial[3]) + abs(trial[4])
    if norm <= Constants.SMALL
        return SVector{4,T}(defdir == 1 ? one(T) : zero(T), defdir == 2 ? one(T) : zero(T),
            defdir == 3 ? one(T) : zero(T), defdir == 4 ? one(T) : zero(T))
    else
        return SVector{4,T}(trial[1], trial[2], trial[3], trial[4])
    end
end

"""
    normalize_vector(vcon, Gcov)

Rescale `vcon` so that `|v.v| = 1` under the metric `Gcov`.

# Arguments
- `vcon`: Vector to be normalized.
- `Gcov`: Covariant metric tensor used for normalization.

# Returns
- The normalized vector.
"""
function normalize_vector(vcon, Gcov)
    norm = 0.0
    for k in 1:4
        for l in 1:4
            norm += vcon[k] * vcon[l] * Gcov[k, l]
        end
    end

    norm = sqrt(abs(norm))
    T = eltype(vcon)
    return SVector{4,T}(vcon[1] / norm, vcon[2] / norm, vcon[3] / norm, vcon[4] / norm)
end

"""
    project_out(vcona, vconb, Gcov)

Project out the component of `vcona` along `vconb` under the metric
`Gcov`. The result is orthogonal to `vconb`.

# Arguments
- `vcona`: Vector to be projected.
- `vconb`: Vector to project out.
- `Gcov`: Covariant metric tensor used for the projection.

# Returns
- `vcona`, with its component along `vconb` removed.
"""
function project_out(vcona, vconb, Gcov)
    vconb_sq = 0.0
    for k in 1:4
        for l in 1:4
            vconb_sq += vconb[k] * vconb[l] * Gcov[k, l]
        end
    end

    adotb = 0.0
    for k in 1:4
        for l in 1:4
            adotb += vcona[k] * vconb[l] * Gcov[k, l]
        end
    end
    fac = adotb / vconb_sq
    T = eltype(vcona)
    return SVector{4,T}(vcona[1] - vconb[1] * fac, vcona[2] - vconb[2] * fac,
        vcona[3] - vconb[3] * fac, vcona[4] - vconb[4] * fac)
end

"""
    levi_civita(i, j, k, l)

Compute the (4-index) Levi-Civita symbol.

# Arguments
- `i`, `j`, `k`, `l`: Indices for which the Levi-Civita symbol is computed.

# Returns
- `0` if any two indices coincide, otherwise `+1`/`-1` depending on the
  permutation parity.
"""
function levi_civita(i::Int, j::Int, k::Int, l::Int)
    return (i == j || i == k || i == l || j == k || j == l || k == l) ? 0 : sign((i - j) * (k - l))
end

"""
    check_handedness(Econ, Gcov)

Check the handedness of a tetrad basis.

# Arguments
- `Econ`: Tetrad basis vectors in covariant form.
- `Gcov`: Covariant metric tensor.

# Returns
- A tuple `(flag, dot_var)`, where `flag` is `1` if `Gcov` is singular
  (and `0` otherwise), and `dot_var` is `+1` for a right-handed basis and
  `-1` for a left-handed one.
"""
function check_handedness(Econ, Gcov)
    T = eltype(Econ)
    g = det(Gcov)
    if abs(g) < 1e-14
        return (1, zero(T))
    end
    g = sqrt(abs(g))

    dot_var = g * det(Econ)
    return (0, dot_var)
end

end
