"""
Small linear-algebra helpers used when building tetrads.
"""
module Utils

using StaticArrays
using ..Constants
using ..Metrics

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
    Econ = MVector{4,eltype(trial)}(undef)
    norm = sum(abs.(trial[2:4]))
    for k in 1:4
        if norm <= Constants.SMALL
            Econ[k] = (k == defdir) ? 1.0 : 0.0
        else
            Econ[k] = trial[k]
        end
    end
    return Econ
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
    vcon_out = copy(vcon)

    norm = 0.0
    for k in 1:4
        for l in 1:4
            norm += vcon[k] * vcon[l] * Gcov[k, l]
        end
    end

    norm = sqrt(abs(norm))
    for k in 1:4
        vcon_out[k] /= norm
    end
    return vcon_out
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
    vcona_out = copy(vcona)
    for k in 1:4
        vcona_out[k] -= vconb[k] * adotb / vconb_sq
    end
    return vcona_out
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
    g = Metrics.gdet_func(Gcov)
    if g < 0.0
        @warn "Encountered singular gcov checking handedness!"
        return (1, 0.0)
    end
    dot_var = zero(eltype(Econ))
    for i in 1:4, j in 1:4, l in 1:4, k in 1:4
        dot_var += g * levi_civita(i - 1, j - 1, k - 1, l - 1) * Econ[1, i] * Econ[2, j] * Econ[3, k] * Econ[4, l]
    end

    return (0, dot_var)
end

end
