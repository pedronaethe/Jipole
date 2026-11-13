"""
Reader for native KHARMA dumps (Parthenon `.phdf` HDF5 files), following `ipole`'s
`model/kharma/model.c`.

Every emission, geodesic, output and GPU routine written for `Iharm`
works on KHARMA dumps unchanged. Only the reading differs: KHARMA stores its grid and
coordinate parameters in the input deck saved inside the dump, and its fluid variables
 split into meshblocks, which are kept as they are.
"""
module Kharma

using HDF5
using Printf
using StaticArrays
using ..Constants
using ..Metrics
using ..Iharm
using ..Grid

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
    params = Iharm.IharmParamsBuilder(MBH; th_beg, Rlow, Rhigh, beta_crit, sigma_cut, sigma_cut_high, slow_light, M_unit)

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
        end
    end

   return Iharm.IharmParams(params, cstopx_2)
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

"""
    load_data(filename, Rhigh, model; advance_path! = nothing)

Read a KHARMA dump and build its IharmData snapshot, keeping the meshblocks the dump is
stored in (refined or not). The mesh is built from the block face coordinates saved in the
dump, and each block's cells go into the inside of a block with one ghost layer;
build_data then fills the ghosts from the neighbouring blocks.
"""
function load_data(filename::String, Rhigh, model::Iharm.IharmParams; advance_path!::Union{Nothing,Function}=nothing)
    println("Loading data from '$filename' into 'Kharma' module...")
    isfile(filename) || error("File not found: $filename")

    t, fields, mesh = h5open(filename, "r") do file
        info = file["Info"]
        nblocks = Int(read_attribute(info, "NumMeshBlocks"))
        nb = Tuple(Int.(read_attribute(info, "MeshBlockSize")))
        if haskey(attributes(info), "IncludesGhost") && read_attribute(info, "IncludesGhost") != 0
            error("KHARMA dump was written with ghost zones, which is not supported")
        end

        # Faces of every block in native coordinates, size (cells + 1, nblocks) in each direction.
        # They define the mesh, so refinement levels never have to be read.
        xf, yf, zf = read(file["Locations"]["x"]), read(file["Locations"]["y"]), read(file["Locations"]["z"])
        size(xf) == (nb[1] + 1, nblocks) && size(yf) == (nb[2] + 1, nblocks) && size(zf) == (nb[3] + 1, nblocks) ||
            error("KHARMA block faces have sizes $(size(xf)), $(size(yf)), $(size(zf)); expected (n + 1, $nblocks) for blocks of $nb cells")
        mesh = Grid.BlockMesh([(view(xf, :, mb), view(yf, :, mb), view(zf, :, mb)) for mb in 1:nblocks], nb)

        # The blocks must cover exactly the domain of the input deck that read_header used
        xmax = mesh.xmin .+ size(mesh.lookup) .* mesh.dfine
        all(isapprox.(mesh.xmin, model.startx[2:4]; atol=1e-10)) && all(isapprox.(xmax, model.stopx[2:4]; atol=1e-10)) ||
            error("KHARMA blocks span $(mesh.xmin) to $xmax, but the input deck gives $(model.startx[2:4]) to $(model.stopx[2:4])")

        # Read each dataset once (HDF5 is not thread-safe), then copy every block of every field
        # into the inside of its padded block, in parallel.
        ng = mesh.ng
        fields = [zeros(nb[1] + 2ng[1], nb[2] + 2ng[2], nb[3] + 2ng[3], nblocks) for _ in 1:8]
        rho, u = read(file["prims.rho"]), read(file["prims.u"])
        uvec, B = read(file["prims.uvec"]), read(file["prims.B"])
        sources = ((rho, 1), (u, 1), (uvec, 1), (uvec, 2), (uvec, 3), (B, 1), (B, 2), (B, 3))
        Threads.@threads for fm in CartesianIndices((8, nblocks))
                        f, mb = Tuple(fm)
            A, c = sources[f]
            view(fields[f], ng[1]+1:ng[1]+nb[1], ng[2]+1:ng[2]+nb[2], ng[3]+1:ng[3]+nb[3], mb) .= block_view(A, c, mb, nb)
        end

        Float64(read_attribute(info, "Time")), fields, mesh
    end

    data = Iharm.build_data(t, fields..., mesh, Rhigh, model)
    println("All primitives successfully loaded: $(size(data.RHO, 4)) blocks of size ", size(data.RHO)[1:3])

    advance_path! === nothing || advance_path!()
    return data
end

end
