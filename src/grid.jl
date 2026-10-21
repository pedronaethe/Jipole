"""
GRMHD simulation grid indexing and interpolation (used by `Iharm`).
"""
module Grid

using StaticArrays
using Adapt

export x_in_domain, interp_scalar, interp_scalar_time, ZoneLoc, locate

"""
Mesh block structure necessary for AMR simulations.
"""
struct BlockMesh{AI<:AbstractArray{Int32,3}, AF<:AbstractMatrix{Float64}}
    lookup::AI                  # finest-level block grid -> block id
    bstart::AF                  # (3, nblocks) native coord of each block's first interior face
    bdx::AF                     # (3, nblocks) cell size of each block
    xmin::SVector{3,Float64}    # start of the whole mesh
    dfine::SVector{3,Float64}   # extent of one finest-level block (= one lookup entry)
    nb::SVector{3,Int}          # interior cells per block
    ng::SVector{3,Int}          # ghost cells per side (0 in a direction with a single cell, e.g. X3 in 2D)
end

Adapt.@adapt_structure BlockMesh

"""
    uniform_mesh(model)

Basically transform an uniform grid into a mesh block structure. This is used for the uniform grid case, where the mesh is just one block.
It makes a single block with dimensions `model.N1`, `model.N2`, `model.N3` and the faces are just the startx and stopx of the model.
"""
function uniform_mesh(model)
    N = (model.N1, model.N2, model.N3)
    faces = ntuple(d -> model.startx[d+1] .+ (0:N[d]) .* model.dx[d+1], 3)
    return BlockMesh([faces], N)
end

"""
BlockMesh(xfaces, nb)

This constructor builds the Mesh mapping. Given any point, it returns which block it is in. 
"""
function BlockMesh(xfaces, nb)
    # Number of blocks, where xfaces is just each block’s three arrays of faces
    nblocks = length(xfaces)

    #nb is the number of cells per block which is the same for all blocks.
    nb = SVector{3,Int}(nb)
    # ng is the number of ghost cells per block. One ghost layer per side in every direction that has more than one cell.
    ng = SVector{3,Int}(ntuple(d -> nb[d] > 1 ? 1 : 0, 3))
    bstart = zeros(3, nblocks)
    bend = zeros(3, nblocks)
    bdx = zeros(3, nblocks)
    # get where the block starts, where it ends and it's size in each direction.
    # Later, locate uses bstart and bdx to find the cell inside a block
    for b in 1:nblocks, d in 1:3
        f = xfaces[b][d]
        bstart[d, b] = f[1]
        bend[d, b] = f[end]
        bdx[d, b] = (f[end] - f[1]) / nb[d]
    end

    # The whole mesh start and end.
    xmin = SVector{3,Float64}(ntuple(d -> minimum(bstart[d, :]), 3))
    xmax = SVector{3,Float64}(ntuple(d -> maximum(bend[d, :]), 3))
    #finest block width.
    dfine = SVector{3,Float64}(ntuple(d -> minimum(bend[d, :] .- bstart[d, :]), 3))
    nlook = ntuple(d -> round(Int, (xmax[d] - xmin[d]) / dfine[d]), 3)

    lookup = zeros(Int32, nlook)
    for b in 1:nblocks
        # for each block, find which slots it covers from lo to hi.
        lo = ntuple(d -> round(Int, (bstart[d, b] - xmin[d]) / dfine[d]) + 1, 3)
        hi = ntuple(d -> round(Int, (bend[d, b] - xmin[d]) / dfine[d]), 3)
        region = view(lookup, lo[1]:hi[1], lo[2]:hi[2], lo[3]:hi[3])
        all(==(0), region) || error("block $b overlaps another block")
        region .= b
        all(d -> isapprox((bstart[d, b] - xmin[d]) / dfine[d], lo[d] - 1; atol=1e-6) &&
                   isapprox((bend[d, b] - xmin[d]) / dfine[d], hi[d]; atol=1e-6), 1:3) ||
              error("block $b is not aligned with the finest-level block grid")
    end
    all(>(0), lookup) || error("blocks do not cover the whole mesh")
    return BlockMesh(lookup, bstart, bdx, xmin, dfine, nb, ng)
end

 """
      find_block(mesh, x1, x2, x3)

  Return the id of the block of `mesh` that contains the point `(x1, x2, x3)` (native
  coordinates), with a single read of `mesh.lookup`, no search.

  Points outside the mesh are clamped to the nearest block on its edge. `locate` relies on
  this for points that are inside the domain (`cstartx`..`cstopx`) but outside the mesh,
  e.g. between `r = 1` and `Rin`. Its stencil clamping then gives the edge value, as before.
  A point exactly on a face between two blocks may go to either one; both contain it.
  """
  @inline function find_block(mesh::BlockMesh, x1, x2, x3)
      nl = size(mesh.lookup)
      l1 = clamp(floor(Int, (x1 - mesh.xmin[1]) / mesh.dfine[1]) + 1, 1, nl[1])
      l2 = clamp(floor(Int, (x2 - mesh.xmin[2]) / mesh.dfine[2]) + 1, 1, nl[2])
      l3 = clamp(floor(Int, (x3 - mesh.xmin[3]) / mesh.dfine[3]) + 1, 1, nl[3])
      return Int(@inbounds mesh.lookup[l1, l2, l3])
  end




"""
    ZoneLoc{T}

Where a point is on the mesh, as found by locate: the block it is in (mb), the cells around
it (i, j, k), how far it is from them (del2, del3, del4, with b1 and b2 the matching
weights), and whether it is inside the simulation domain (in_domain).
"""
struct ZoneLoc{T}
    i::Int; j::Int; k::Int
    mb::Int
    del2::T; del3::T; del4::T
    b1::T; b2::T
    in_domain::Bool
end

"""
    locate(X, model, mesh)

Find where the point X is on the block mesh: the block it is in, the cell to interpolate
from (inside that block, ghost cells included) and the interpolation weights.

Phi is wrapped into [0, 2π) as before. Points that are inside the domain but outside
the mesh (e.g. between the horizon and the inner edge of the grid) get the value of the
edge cell, as before.
"""
@inline function locate(X, model, mesh::BlockMesh)
    if x_in_domain(X, model) == 0
        z = zero(eltype(X))
        return ZoneLoc(0, 0, 0, 1, z, z, z, z, z, false)
    end

    # Wrap phi into [0, 2π), as before
    phi = rem(X[4], model.cstopx[4])
    if phi < 0.0
        phi += model.cstopx[4]
    end
    x = (X[2], X[3], phi)

    # Block containing X (one lookup)
    mb = find_block(mesh, x[1], x[2], x[3])

    s = ntuple(3) do d
        # u is the position of X in the block in cell units, counted from the centre of the first interior cell.
        u = (x[d] - mesh.bstart[d, mb]) / mesh.bdx[d, mb] - 0.5
        c = clamp(floor(Int, u), -mesh.ng[d], max(mesh.nb[d] + mesh.ng[d] - 2, -mesh.ng[d]))
        (c + mesh.ng[d] + 1, clamp(u - c, 0.0, 1.0))
    end

    (i, del2), (j, del3), (k, del4) = s
    return ZoneLoc(i, j, k, mb, del2, del3, del4, 1.0 - del2, 1.0 - del3, true)
end             


  """
      fill_ghosts!(A, mesh)

  Fill the ghost cells (the extra layer of cells around each block) of the block array A,
  so that interpolating near the edge of a block never needs data from another block.

  Each ghost cell takes its value from the neighbouring block:
  - same resolution: the value is copied;
  - coarser neighbour: it takes the value of the coarse cell it falls in;
  - finer neighbour: it takes the average of the fine cells it covers.

  At the edges of the whole mesh, ghost cells copy the edge cell in r and theta, and wrap
  around in phi.

  Called once per field when a dump is loaded, before anything else is computed.
  """
function fill_ghosts!(A::AbstractArray{T,4}, mesh::BlockMesh) where {T}
    nb, ng = mesh.nb, mesh.ng
    xmax = mesh.xmin .+ size(mesh.lookup) .* mesh.dfine
    Threads.@threads for I in CartesianIndices(A)
        p = Tuple(I); b = p[4]
        # interior cells are data, skip them
        all(d -> ng[d] < p[d] <= ng[d] + nb[d], 1:3) && continue

        # ghost cell centre, with the outer boundary conditions applied:
        # zero gradient in X1/X2 (clamp onto the edge cell centre), periodic in X3
        xc = ntuple(d -> mesh.bstart[d, b] + (p[d] - ng[d] - 0.5) * mesh.bdx[d, b], 3)
        x = ntuple(3) do d
            d == 3 ? mesh.xmin[3] + mod(xc[3] - mesh.xmin[3], xmax[3] - mesh.xmin[3]) :
                     clamp(xc[d], mesh.xmin[d] + 0.5 * mesh.bdx[d, b], xmax[d] - 0.5 * mesh.bdx[d, b])
        end
        s = find_block(mesh, x...)

        # How many cells of the neighbour to average in each direction (1 unless it is finer),
        # and the first of them, as an index into the padded array
        n = ntuple(d -> mesh.bdx[d, b] / mesh.bdx[d, s] > 1.5 ? round(Int, mesh.bdx[d, b] / mesh.bdx[d, s]) : 1, 3)
        lo = ntuple(3) do d
            c = n[d] == 1 ? floor(Int, (x[d] - mesh.bstart[d, s]) / mesh.bdx[d, s]) :
                            floor(Int, (x[d] - 0.5 * mesh.bdx[d, b] - mesh.bstart[d, s]) / mesh.bdx[d, s] + 0.5)
            ng[d] + clamp(c, 0, nb[d] - n[d]) + 1
        end

        acc = zero(T)
        for k in lo[3]:lo[3]+n[3]-1, j in lo[2]:lo[2]+n[2]-1, i in lo[1]:lo[1]+n[1]-1
            acc += A[i, j, k, s]
        end
        A[I] = acc / (n[1] * n[2] * n[3])
    end
    return A
end

"""
    x_in_domain(X, model)

Check whether the position `X` lies within `model`'s simulation grid
bounds.

# Arguments
- `X`: Position four-vector in internal coordinates.
- `model`: Model parameters, providing the grid bounds (`model.cstartx`,
  `model.cstopx`).

# Returns
- `1` if `X` is within bounds, `0` otherwise.
"""
function x_in_domain(X, model)
    if X[2] < model.cstartx[2] || X[2] > model.cstopx[2] || X[3] < model.cstartx[3] || X[3] > model.cstopx[3]
        return 0
    end
    return 1
end

"""
    interp_scalar(data_arr, zone)

Interpolate a field at a point, using the 8 cells around it.

`zone` (from `locate`) says which block the point is in, which cells surround it, and how
close it is to each of them. Thanks to the ghost cells, all 8 cells are always in the same
block, so the value is just a weighted average of 8 numbers.
"""
@inline function interp_scalar(data_arr::AbstractArray{T,4}, zone::ZoneLoc) where {T}
    mb = zone.mb
    ip1 = zone.i + 1; jp1 = zone.j + 1
    kp1 = min(zone.k + 1, size(data_arr, 3))
    @inbounds interp = data_arr[zone.i, zone.j, zone.k, mb] * zone.b1 * zone.b2 +
                        data_arr[ip1, zone.j, zone.k, mb] * zone.del2 * zone.b2 +
                        data_arr[zone.i, jp1, zone.k, mb] * zone.b1 * zone.del3 +
                        data_arr[ip1, jp1, zone.k, mb] * zone.del2 * zone.del3
    @inbounds interp = interp * (1.0 - zone.del4) + (
        data_arr[zone.i, zone.j, kp1, mb] * zone.b1 * zone.b2 +
        data_arr[ip1, zone.j, kp1, mb] * zone.del2 * zone.b2 +
        data_arr[zone.i, jp1, kp1, mb] * zone.b1 * zone.del3 +
        data_arr[ip1, jp1, kp1, mb] * zone.del2 * zone.del3
    ) * zone.del4
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
function interp_scalar_time(zone::ZoneLoc, dataA, dataB, tfac, slow_light::Bool)
    vA = interp_scalar(dataA, zone)
    if slow_light
        vB = interp_scalar(dataB, zone)
        return tfac * vA + (1.0 - tfac) * vB
    end
    return vA
end


end
