"""
Small linear-algebra helpers used when building tetrads.
"""
module Utils

using StaticArrays
using LinearAlgebra
using ..Constants

export set_econ_from_trial, normalize_vector, project_out, levi_civita, check_handedness

"""
    set_econ_from_trial(defdir, trial)

Copy the trial vector; if its spatial norm is too small, fall back to the
unit vector along `defdir`.

# Arguments
- `defdir`: Direction to default to if `trial` is nearly null.
- `trial`: Trial vector to set the tetrad vector from.

# Returns
- The resulting tetrad basis vector.
"""
function set_econ_from_trial(defdir::Int, trial)
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
    T = promote_type(eltype(vcon), eltype(Gcov))
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
    T = promote_type(eltype(vcona), eltype(vconb), eltype(Gcov))
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
    T = promote_type(eltype(Econ), eltype(Gcov))
    g = det(Gcov)
    if abs(g) < 1e-14
        return (1, zero(T))
    end
    g = sqrt(abs(g))

    dot_var = g * det(Econ)
    return (0, dot_var)
end



"""
    get_config(config, section, key, default)

    Retrieve a configuration value from a nested dictionary, returning a default if the key is not found.
"""

function get_config(config::Dict, section::String, key::String, default)
    if haskey(config, section) && haskey(config[section], key)
        return config[section][key]
    else
        @warn "Parameter [$section].$key not found in configuration. Using default value: $default"
        return default
    end
end

"""
    extract_dump_index(filename)

Pull the run of digits that sits right before a dump file's extension. Returns `nothing` if the filename has
no numeric field.
"""
function extract_dump_index(filename::String)
    m = match(r"(\d+)(?=\.[^.\/]+$)", filename)
    m === nothing && return nothing
    return parse(Int, m.captures[1])
end

"""
    resolve_dump_files(path, t_init, t_final)

Turn dump_filepath into the ordered list of dump files to raytrace.

- If `path` is a single file, that file is the whole list.
- If `path` is a directory, every file in it is scanned and, it tracks which files are t_init and t_final. Files without a
  parseable trailing index are skipped. If nothing falls in range, the
  closest dump to t_init and t_final are used instead.
"""
function resolve_dump_files(path::String, t_init::Int, t_final::Int)
    isfile(path) && return [path]

    isdir(path) || error("dump_filepath '$path' is neither a file nor a directory")

    indexed_files = Tuple{Int,String}[]
    for entry in readdir(path; join=true)
        isfile(entry) || continue
        idx = extract_dump_index(basename(entry))
        idx === nothing && continue
        push!(indexed_files, (idx, entry))
    end

    isempty(indexed_files) && error("No dump files with a numeric index found in directory '$path'")

    sort!(indexed_files; by=first)

    selected = [file for (idx, file) in indexed_files if t_init <= idx <= t_final]

    if isempty(selected)
        closest_idx, closest_file = indexed_files[argmin([abs(idx - t_init) for (idx, _) in indexed_files])]
        @warn "No dump files found with index between $t_init and $t_final. Using the closest match: $closest_file (index $closest_idx)"
        return [closest_file]
    end

    return selected
end

"""
    dump_path_template(example_filepath)

Turn one dump file's path into a format string for the whole
sequence, by replacing its numeric field with the format
`%0Nd` specifier of the same width. Used by slow-light rendering, which walks
the sequence by index (see `Slowlight.update_dump_path`).
"""
function dump_path_template(example_filepath::String)
    m = match(r"(\d+)(?=\.[^.\/]+$)", example_filepath)
    m === nothing && error("Could not find a numeric dump index in '$example_filepath'")
    width = length(m.match)
    prefix = example_filepath[1:m.offset-1]
    suffix = example_filepath[m.offset+width:end]
    return prefix * "%0$(width)d" * suffix
end

"""
    dump_output_filename(template, index)

Insert the dump index into the output filename template. If `index` is `nothing`, returns the template unchanged.
"""
function dump_output_filename(template::String, index::Union{Int,Nothing})
    index === nothing && return template
    base, ext = splitext(template)
    return "$(base)_$(lpad(index, 5, '0'))$(ext)"
end


end
