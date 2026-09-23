"""
Reader for native KHARMA dumps (Parthenon `.phdf` HDF5 files), following `ipole`'s
`model/kharma/model.c`.

Every emission, geodesic, output and GPU routine written for `Iharm`
works on KHARMA dumps unchanged. Only the reading differs: KHARMA stores its grid and
coordinate parameters in the input deck saved inside the dump, and its fluid variables
split into meshblocks that have to be placed back on the global grid.
"""
module Kharma

using HDF5
using Printf
using StaticArrays
using ..Constants
using ..Metrics
using ..Iharm

export is_kharma_dump, read_header, load_data

function is_kharma_dump(filename::String)
    return h5open(filename, "r") do file
        haskey(file, "Info") && haskey(file, "Input")
    end
end


function read_input_deck(file::HDF5.File)
    text = replace(String(read_attribute(file["Input"], "File")), '\0' => ' ')
    deck = Dict{String,Dict{String,String}}()
    block = ""
    for raw_line in split(text, '\n')
        line = strip(first(split(raw_line, '#'; limit=2)))
        isempty(line) && continue
        if startswith(line, '<') && endswith(line, '>')
            block = line[2:end-1]
            get!(deck, block, Dict{String,String}())
        elseif occursin('=', line)
            key, value = strip.(split(line, '='; limit=2))
            get!(deck, block, Dict{String,String}())[key] = value
        end
    end
    return deck
end

function deck_value(deck, block, key, ::Type{T}; default=nothing) where {T}
    value = get(get(deck, block, Dict{String,String}()), key, nothing)
    if value === nothing
        default === nothing && error("KHARMA input deck has no `$key` in <$block>")
        return convert(T, default)
    end
    return T === String ? String(value) : parse(T, value)
end

function read_header(filename::String, MBH; th_beg=1.74e-2, Rlow=1.0, Rhigh=20.0, beta_crit=1.0, sigma_cut=1.0, sigma_cut_high=-1.0, slow_light=false, M_unit=3.e26)
    println("Initializing grid from KHARMA dump: $filename")

    params = Iharm.IharmParamsBuilder()
    params.th_beg = th_beg
    params.Rlow = Rlow
    params.Rhigh = Rhigh
    params.beta_crit = beta_crit
    params.sigma_cut = sigma_cut
    params.sigma_cut_high = sigma_cut_high
    params.slow_light = slow_light

    params.L_unit = Constants.GNEWT * MBH * Constants.MSUN / Constants.CL^2
    params.T_unit = params.L_unit / Constants.CL
    params.MBH = MBH
    params.M_unit = M_unit

    cstopx_2 = 0.0

    h5open(filename, "r") do file
        deck = read_input_deck(file)

        # KHARMA accepts both the short and the long name of each coordinate transform
        transform = lowercase(deck_value(deck, "coordinates", "transform", String))
        if transform in ("mks", "modified")
            params.metric = Metrics.METRIC_MKS
            cstopx_2 = 1.0
            @printf(stderr, "Using Modified Kerr-Schild coordinates MKS\n")
        elseif transform in ("fmks", "funky")
            params.metric = Metrics.METRIC_FMKS
            cstopx_2 = 1.0
            @printf(stderr, "Using Funky Modified Kerr-Schild coordinates FMKS\n")
        elseif transform in ("eks", "exponential")
            params.metric = Metrics.METRIC_EKS
            cstopx_2 = π
            @printf(stderr, "Using Kerr-Schild coordinates with exponential radial coordinate\n")
        else
            error("KHARMA coordinate transform '$transform' is not supported (use mks, fmks or eks).")
        end

        # Grid: the global mesh, not the meshblocks
        params.N1 = deck_value(deck, "parthenon/mesh", "nx1", Int)
        params.N2 = deck_value(deck, "parthenon/mesh", "nx2", Int)
        params.N3 = deck_value(deck, "parthenon/mesh", "nx3", Int)
        x1min = deck_value(deck, "parthenon/mesh", "x1min", Float64)
        x1max = deck_value(deck, "parthenon/mesh", "x1max", Float64)
        x2min = deck_value(deck, "parthenon/mesh", "x2min", Float64)
        x2max = deck_value(deck, "parthenon/mesh", "x2max", Float64)
        x3min = deck_value(deck, "parthenon/mesh", "x3min", Float64)
        x3max = deck_value(deck, "parthenon/mesh", "x3max", Float64)
        params.startx[2] = x1min
        params.startx[3] = x2min
        params.startx[4] = x3min
        params.dx[2] = (x1max - x1min) / params.N1
        params.dx[3] = (x2max - x2min) / params.N2
        params.dx[4] = (x3max - x3min) / params.N3

        # Fluid and electrons
        params.gam = deck_value(deck, "GRMHD", "gamma", Float64)
        if deck_value(deck, "electrons", "on", Bool; default=false)
            params.game = deck_value(deck, "electrons", "gamma_e", Float64; default=params.game)
            params.gamp = deck_value(deck, "electrons", "gamma_p", Float64; default=params.gamp)
            @printf(stderr, "KHARMA dump has electron physics; using the Rlow/Rhigh model instead of its electron entropies\n")
        end
        params.ELECTRONS = 2
        params.Te_unit = params.Thetae_unit

        # Coordinate parameters. The radial coordinate is log(r) in all three systems, so the
        # mesh bounds give Rin/Rout if the deck does not list them.
        params.a = deck_value(deck, "coordinates", "a", Float64)
        params.Rin = deck_value(deck, "coordinates", "r_in", Float64; default=exp(x1min))
        params.Rout = deck_value(deck, "coordinates", "r_out", Float64; default=exp(x1max))
        if params.metric == Metrics.METRIC_EKS
            @printf(stderr, "eKS parameters a: %f Rin: %f Rout: %f\n", params.a, params.Rin, params.Rout)
        else
            params.hslope = deck_value(deck, "coordinates", "hslope", Float64)
            @printf(stderr, "MKS parameters a: %f hslope: %f Rin: %f Rout: %f\n", params.a, params.hslope, params.Rin, params.Rout)
        end
        if params.metric == Metrics.METRIC_FMKS
            params.mks_smooth = deck_value(deck, "coordinates", "mks_smooth", Float64)
            params.poly_xt = deck_value(deck, "coordinates", "poly_xt", Float64)
            params.poly_alpha = deck_value(deck, "coordinates", "poly_alpha", Float64)
            params.poly_norm = 0.5 * π * 1.0 / (1.0 + 1.0 / (params.poly_alpha + 1.0) * 1.0 / (params.poly_xt^params.poly_alpha))
            @printf(stderr, "FMKS parameters poly_xt: %f poly_alpha: %f mks_smooth: %f poly_norm: %f\n",
                params.poly_xt, params.poly_alpha, params.mks_smooth, params.poly_norm)
        end
    end

    # From here on, exactly as Iharm.read_header
    params.rmax_geo = min(params.rmax_geo, params.Rout)
    params.rmin_geo = max(params.rmin_geo, params.Rin)

    params.stopx = MVector{4,Float64}(
        1.0,
        params.startx[2] + params.N1 * params.dx[2],
        params.startx[3] + params.N2 * params.dx[3],
        params.startx[4] + params.N3 * params.dx[4]
    )

    params.cstartx = MVector{4,Float64}(0.0, 0.0, 0.0, 0.0)
    params.cstopx = MVector{4,Float64}(0.0, 0.0, cstopx_2, 2 * π)
    if params.metric != Metrics.METRIC_EKS
        params.cstopx[2] = log(params.Rout)
    end

    @printf(stderr, "Grid start (startx): %.15e, %.15e, %.15e stop (stopx): %.15e, %.15e, %.15e\n",
        params.startx[2], params.startx[3], params.startx[4], params.stopx[2], params.stopx[3], params.stopx[4])
    @printf(stderr, "grid dx: %.15e, %.15e, %.15e\n", params.dx[2], params.dx[3], params.dx[4])

    params.RHO_unit = params.M_unit / params.L_unit^3
    params.U_unit = params.RHO_unit * Constants.CL^2
    params.B_unit = Constants.CL * sqrt(4 * π * params.RHO_unit)

    return Iharm.IharmParams(params)
end

function read_block_locations(file::HDF5.File, nblocks)
    if haskey(file, "Blocks") && haskey(file["Blocks"], "loc.lx123")
        locs = read(file["Blocks"]["loc.lx123"])
    elseif haskey(file, "LogicalLocations")           # older Parthenon outputs
        locs = read(file["LogicalLocations"])
    else
        error("KHARMA dump has no meshblock locations (Blocks/loc.lx123)")
    end
    size(locs) == (3, nblocks) || error("Meshblock locations have size $(size(locs)), expected (3, $nblocks)")

    if haskey(file, "Blocks") && haskey(file["Blocks"], "loc.level")
        levels = read(file["Blocks"]["loc.level"])
        all(==(first(levels)), levels) || error("KHARMA dump uses mesh refinement, which is not supported")
    end
    return Int.(locs)
end

function block_view(A, c, mb, nb)
    if size(A)[1:3] == nb
        return ndims(A) == 4 ? view(A, :, :, :, mb) : view(A, :, :, :, c, mb)
    elseif size(A)[2:4] == nb
        return view(A, c, :, :, :, mb)
    else
        error("KHARMA dataset of size $(size(A)) does not match meshblock size $nb")
    end
end

function scatter_blocks!(dest, A, c, locs, nb)
    nb1, nb2, nb3 = nb
    Threads.@threads for mb in axes(locs, 2)
        i0 = locs[1, mb] * nb1
        j0 = locs[2, mb] * nb2
        k0 = locs[3, mb] * nb3
        dest[i0+1:i0+nb1, j0+1:j0+nb2, k0+1:k0+nb3] .= block_view(A, c, mb, nb)
    end
    return dest
end

function load_data(filename::String, Rhigh, model::Iharm.IharmParams; advance_path!::Union{Nothing,Function}=nothing)
    println("Loading data from '$filename' into 'Kharma' module...")
    isfile(filename) || error("File not found: $filename")

    t, fields = h5open(filename, "r") do file
        info = file["Info"]
        nblocks = Int(read_attribute(info, "NumMeshBlocks"))
        nb = Tuple(Int.(read_attribute(info, "MeshBlockSize")))
        if haskey(attributes(info), "IncludesGhost") && read_attribute(info, "IncludesGhost") != 0
            error("KHARMA dump was written with ghost zones, which is not supported")
        end
        nblocks * prod(nb) == model.N1 * model.N2 * model.N3 ||
            error("$nblocks meshblocks of size $nb do not tile the $(model.N1)×$(model.N2)×$(model.N3) mesh")

        locs = read_block_locations(file, nblocks)
        fields = [Array{Float64}(undef, model.N1, model.N2, model.N3) for _ in 1:8]

        # Read each dataset once (HDF5 is not thread-safe); placing the blocks is threaded.
        scatter_blocks!(fields[1], read(file["prims.rho"]), 1, locs, nb)
        scatter_blocks!(fields[2], read(file["prims.u"]), 1, locs, nb)
        uvec = read(file["prims.uvec"])
        B = read(file["prims.B"])
        for c in 1:3
            scatter_blocks!(fields[2+c], uvec, c, locs, nb)
            scatter_blocks!(fields[5+c], B, c, locs, nb)
        end

        Float64(read_attribute(info, "Time")), fields
    end

    data = Iharm.build_data(t, fields..., Rhigh, model)
    println("All primitives successfully loaded. Dimensions: ", size(data.RHO))

    advance_path! === nothing || advance_path!()
    return data
end

end
