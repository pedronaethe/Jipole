"""
GRMHD simulation-data emission model (`ipole`'s "iharm" model), reading
dumps in `iharm3d`/`KHARMA` HDF5 format.
"""
module Iharm

using HDF5
using Printf
using StaticArrays
using ForwardDiff
using Adapt
using ..Constants
using ..AbstractModels
using ..Coordinates
using ..Metrics
using ..Camera
using ..MaxwellJuettnerPol
using ..Grid
using ..Radiation
using ..MaxwellJuettner
using ..Imaging
using ..Polarization
using ..ImagingPol

export IharmParams, IharmParamsBuilder, IharmData, read_header, load_data,
    compute_accretion_diagnostics, grmhd_context, calculate_gradients

const VALID_PRIMS = ["RHO", "UU", "U1", "U2", "U3", "B1", "B2", "B3"]
const USE_GEODESIC_SIGMACUT = true
const ELECTRONS_TFLUID = 3
const USE_FIXED_TPTE = false
const USE_MIXED_TPTE = true

"""
A single GRMHD simulation snapshot: fluid primitives on the simulation
grid, plus the derived electron/magnetic-field quantities used by the
radiative transfer, and the block mesh they live on.

Every field is stored as blocks: `field[i, j, k, mb]`, with one layer of ghost cells
around each block (see [`Grid.BlockMesh`](@ref) and [`Grid.fill_ghosts!`](@ref)). A
uniform iharm3d dump is a mesh with a single block; KHARMA dumps keep their own blocks.

There are two array types: `Araw` for the raw fluid primitives
(`RHO`/`UU`/`U1`-`U3`/`B1`-`B3`), which are always `Float64`, and `Ader` for the derived
quantities (`ne`/`b`/`θe`/`sigma`). They are separate so that, when
differentiating with respect to M_unit, Rhigh, etc. (see [`calculate_gradients`](@ref)),
only the derived quantities become `ForwardDiff.Dual`; the primitives don't depend on
those parameters and stay `Float64` (a `Dual{Float64,2}` takes 24 bytes instead of 8).
On the GPU, all arrays (and the mesh) are `CuArray`s.
"""
struct IharmData{Traw, Araw<:AbstractArray{Traw,4}, Tder, Ader<:AbstractArray{Tder,4}, M<:Grid.BlockMesh}
    t::Float64
    RHO::Araw
    UU::Araw
    U1::Araw
    U2::Araw
    U3::Araw
    B1::Araw
    B2::Araw
    B3::Araw
    ne::Ader
    b::Ader
    θe::Ader
    sigma::Ader
    mesh::M
end

Adapt.@adapt_structure IharmData

"""
Parameters for the [`Iharm`](@ref) GRMHD emission model, immutable and
GPU-safe (`SVector` fields, no heap-allocated members) so it can be used
directly both by the CPU raytracer and as a CUDA kernel argument.

Built once by [`read_header`](@ref) from an [`IharmParamsBuilder`](@ref).

Parametric over `T` (the type of `M_unit`/`RHO_unit`/`U_unit`/`B_unit`
only, every other field stays `Float64` regardless of `T`, since they
don't depend on `M_unit`). This exists so a `ForwardDiff.Dual`-valued
`M_unit` (and the `RHO_unit`/`U_unit`/`B_unit` derived from it) can flow
through, for `dI/dM_unit`; `read_header` always builds an ordinary
`IharmParams{Float64}` -- only code that explicitly differentiates
w.r.t. `M_unit` constructs an `IharmParams{<:ForwardDiff.Dual}`. Function
signatures elsewhere that just say `model::IharmParams` (no explicit
`{T}`) continue to match either case unchanged.
"""
struct IharmParams{T} <: AbstractModel
    metric::Int
    ELECTRONS::Int
    RADIATION::Int

    gam::Float64
    game::Float64
    gamp::Float64
    Te_unit::Float64
    Thetae_unit::Float64

    mu_i::Float64
    mu_e::Float64
    mu_tot::Float64
    Ne_factor::Float64

    M_unit::T
    T_unit::T
    L_unit::T
    MBH::T
    tp_over_te::T

    RHO_unit::T
    U_unit::T
    B_unit::T

    a::T
    hslope::Float64
    Rin::Float64
    Rout::Float64
    poly_xt::Float64
    poly_alpha::Float64
    mks_smooth::Float64
    poly_norm::Float64

    mks3R0::Float64
    mks3H0::Float64
    mks3MY1::Float64
    mks3MY2::Float64
    mks3MP0::Float64

    N1::Int
    N2::Int
    N3::Int

    dx::SVector{4,Float64}
    startx::SVector{4,Float64}
    stopx::SVector{4,Float64}
    cstartx::SVector{4,Float64}
    cstopx::SVector{4,Float64}

    rmin_geo::Float64
    rmax_geo::Float64

    th_beg::T
    Rlow::T
    Rhigh::T
    beta_crit::T
    sigma_cut::T
    sigma_cut_high::T

    slow_light::Bool
end


"""
Mutable, `MVector`-based scratch struct used only while
[`read_header`](@ref) is incrementally parsing a dump file's header
(most fields mirror `ipole`'s C `Params`/`Grid` globals). Once parsing is
complete, `read_header` converts it to the immutable, GPU-safe
[`IharmParams`](@ref) that the rest of the codebase uses.
"""
mutable struct IharmParamsBuilder{T} <: AbstractModel
    metric::Int
    ELECTRONS::Int
    RADIATION::Int

    gam::Float64
    game::Float64
    gamp::Float64
    Te_unit::Float64
    Thetae_unit::Float64

    mu_i::Float64
    mu_e::Float64
    mu_tot::Float64
    Ne_factor::Float64

    M_unit::T
    T_unit::T
    L_unit::T
    MBH::T
    tp_over_te::T

    RHO_unit::T
    U_unit::T
    B_unit::T

    a::T
    hslope::Float64
    Rin::Float64
    Rout::Float64
    poly_xt::Float64
    poly_alpha::Float64
    mks_smooth::Float64
    poly_norm::Float64

    mks3R0::Float64
    mks3H0::Float64
    mks3MY1::Float64
    mks3MY2::Float64
    mks3MP0::Float64

    N1::Int64
    N2::Int64
    N3::Int64
    dx::MVector{4,Float64}
    startx::MVector{4,Float64}
    stopx::MVector{4,Float64}
    cstartx::MVector{4,Float64}
    cstopx::MVector{4,Float64}

    rmin_geo::Float64
    rmax_geo::Float64

    th_beg::T
    Rlow::T
    Rhigh::T
    beta_crit::T
    sigma_cut::T
    sigma_cut_high::T
    slow_light::Bool
end

"""
    IharmParamsBuilder()

Default-constructed [`IharmParamsBuilder`](@ref), used internally by
[`read_header`](@ref) before the dump file's header is parsed into it.
"""
function IharmParamsBuilder()
    return IharmParamsBuilder(0, 0, 0,
        5 / 3, 4 / 3, 5 / 3, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 3.0,
        1.0, 1.0, 1.0,
        0.0, 1.0, 0.0, 100.0,
        0.82, 14.0, 0.5, 1.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
        0, 0, 0,
        MVector{4,Float64}(0.0, 0.0, 0.0, 0.0),
        MVector{4,Float64}(0.0, 0.0, 0.0, 0.0),
        MVector{4,Float64}(0.0, 0.0, 0.0, 0.0),
        MVector{4,Float64}(0.0, 0.0, 0.0, 0.0),
        MVector{4,Float64}(0.0, 0.0, 0.0, 0.0),
        1.0, 100.0,
        1.74e-2, 1.0, 20.0, 1.0, 1.0, -1.0, false)
end

"""
    IharmParamsBuilder(MBH; th_beg, Rlow, Rhigh, beta_crit, sigma_cut, sigma_cut_high, slow_light, M_unit)

Start a header: a default builder with the user's plasma and cut parameters, the
black hole mass and mass unit, and the length and time units set by `MBH`.
`read_header` then fills in what the dump file says.
"""
function IharmParamsBuilder(MBH; th_beg, Rlow, Rhigh, beta_crit, sigma_cut, sigma_cut_high, slow_light, M_unit)
    p = IharmParamsBuilder()
    p.th_beg = th_beg
    p.Rlow = Rlow
    p.Rhigh = Rhigh
    p.beta_crit = beta_crit
    p.sigma_cut = sigma_cut
    p.sigma_cut_high = sigma_cut_high
    p.slow_light = slow_light
    p.L_unit = Constants.GNEWT * MBH * Constants.MSUN / Constants.CL^2
    p.T_unit = p.L_unit / Constants.CL
    p.MBH = MBH
    p.M_unit = M_unit
    return p
end

"""
    IharmParams(p::IharmParamsBuilder, cstopx_2)

Finish a header filled by `read_header` (iharm3d or KHARMA dump) and freeze it into
the immutable, GPU-safe [`IharmParams`](@ref): the FMKS normalization, the
geodesic radial range, the grid end, the coordinate bounds (`cstopx_2` is the upper
bound of X2 for this metric), and the density, energy and field units from
`M_unit` and `L_unit`.
"""
function IharmParams(p::IharmParamsBuilder, cstopx_2)
    if p.metric == Metrics.METRIC_FMKS
        p.poly_norm = 0.5 * π * 1.0 / (1.0 + 1.0 / (p.poly_alpha + 1.0) * 1.0 / (p.poly_xt^p.poly_alpha))
        @printf(stderr, "FMKS parameters poly_xt: %f poly_alpha: %f mks_smooth: %f poly_norm: %f\n",
            p.poly_xt, p.poly_alpha, p.mks_smooth, p.poly_norm)
    end

    p.rmax_geo = min(p.rmax_geo, p.Rout)
    p.rmin_geo = max(p.rmin_geo, p.Rin)

    p.stopx = MVector{4,Float64}(
        1.0,
        p.startx[2] + p.N1 * p.dx[2],
        p.startx[3] + p.N2 * p.dx[3],
        p.startx[4] + p.N3 * p.dx[4]
    )

    p.cstartx = MVector{4,Float64}(0.0, 0.0, 0.0, 0.0)
    p.cstopx = MVector{4,Float64}(0.0, 0.0, cstopx_2, 2 * π)
    if p.metric != Metrics.METRIC_BHACMKS && p.metric != Metrics.METRIC_EKS
        p.cstopx[2] = log(p.Rout)
    end

    @printf(stderr, "Grid start (startx): %.15e, %.15e, %.15e stop (stopx): %.15e, %.15e, %.15e\n",
        p.startx[2], p.startx[3], p.startx[4], p.stopx[2], p.stopx[3], p.stopx[4])
    @printf(stderr, "grid dx: %.15e, %.15e, %.15e\n", p.dx[2], p.dx[3], p.dx[4])

    p.RHO_unit = p.M_unit / p.L_unit^3
    p.U_unit = p.RHO_unit * Constants.CL^2
    p.B_unit = Constants.CL * sqrt(4 * π * p.RHO_unit)

    return IharmParams(p.metric, p.ELECTRONS, p.RADIATION, p.gam, p.game, p.gamp, p.Te_unit, p.Thetae_unit, p.mu_i, p.mu_e, p.mu_tot, p.Ne_factor,
        p.M_unit, p.T_unit, p.L_unit, p.MBH, p.tp_over_te, p.RHO_unit, p.U_unit, p.B_unit, p.a, p.hslope, p.Rin, p.Rout, p.poly_xt, p.poly_alpha, p.mks_smooth, p.poly_norm, p.mks3R0,
        p.mks3H0, p.mks3MY1, p.mks3MY2, p.mks3MP0, p.N1, p.N2, p.N3, SVector(p.dx), SVector(p.startx), SVector(p.stopx), SVector(p.cstartx), SVector(p.cstopx), p.rmin_geo, p.rmax_geo, p.th_beg,
        p.Rlow, p.Rhigh, p.beta_crit, p.sigma_cut, p.sigma_cut_high, p.slow_light)
end


"""
    build_data(t, rho, uu, u1, u2, u3, b1, b2, b3, mesh, Rhigh, model)

Build an IharmData snapshot from the fluid primitives read from a dump.

The primitives come in as block arrays with only the inside of each block filled. This
fills their ghost cells, computes the magnetic field strength b in every cell (from the
metric at the cell centre), fills the ghost cells of b too, and then computes the other
derived quantities (electron density, temperature, sigma, beta).

Used by both readers, Iharm.load_data and Kharma.load_data, which only differ in how they
read the primitives from the file.
"""
function build_data(t, rho, uu, u1, u2, u3, b1, b2, b3, mesh, Rhigh, model::IharmParams)
    # Ghost cells of the primitives, from the neighbouring blocks (once, at load time)
    for A in (rho, uu, u1, u2, u3, b1, b2, b3)
        Grid.fill_ghosts!(A, mesh)
    end

    data = IharmData(Float64(t), rho, uu, u1, u2, u3, b1, b2, b3,
        zeros(size(rho)), zeros(size(rho)), zeros(size(rho)), zeros(size(rho)), mesh)

    # Magnetic field strength b in the inside cells of every block. The metric depends only on
    # X1 and X2, so it is computed once per (i, j) of each block, then used for every k.
    # Static vectors/matrices only, so nothing is allocated per cell.
    nb, ng = mesh.nb, mesh.ng
    Threads.@threads for jb in CartesianIndices((nb[2], size(rho, 4)))
        jj, mb = Tuple(jb)
        j = jj + ng[2]
        for ii in 1:nb[1]
            i = ii + ng[1]
            X = SVector(0.0,
                mesh.bstart[1, mb] + (ii - 0.5) * mesh.bdx[1, mb],
                mesh.bstart[2, mb] + (jj - 0.5) * mesh.bdx[2, mb],
                mesh.bstart[3, mb] + 0.5 * mesh.bdx[3, mb])
            gcov = Metrics.gcov_func(X, model.a, model)
            gcon = Metrics.gcon_func(gcov)

            for kk in 1:nb[3]
                k = kk + ng[3]    
                 U = SVector(u1[i, j, k, mb], u2[i, j, k, mb], u3[i, j, k, mb])
                B = SVector(b1[i, j, k, mb], b2[i, j, k, mb], b3[i, j, k, mb])
                _, _, bcon, bcov = fluid_fourv(Float64, U, B, gcov, gcon)

                bsq = 0.0
                for l in 1:Constants.NDIM
                    bsq += bcov[l] * bcon[l]
                end
                data.b[i, j, k, mb] = sqrt(bsq) * model.B_unit
            end
        end
    end

    # Ghost cells of b like the primitives (not recomputed), so the edges of the mesh behave as before
    Grid.fill_ghosts!(data.b, mesh)

    init_physical_quantities([data], 1, model, Rhigh)
    return data
end


"""
    read_header(filename, MBH; th_beg=1.74e-2, Rlow=1.0, Rhigh=20.0, beta_crit=1.0, sigma_cut=1.0, sigma_cut_high=-1.0, slow_light=false, M_unit=3.e26)

Read a GRMHD dump file's header, returning the populated
[`IharmParams`](@ref).

The electron-temperature-model parameters (`th_beg`, `Rlow`,
`beta_crit`, `sigma_cut`, `sigma_cut_high`) are run configuration, not
recorded in the dump file, so they're supplied here (with `ipole`'s usual
defaults). `L_unit`/`T_unit`/`RHO_unit`/`U_unit`/`B_unit` are derived from
the black hole mass `MBH` and `M_unit`.

# Arguments
- `filename`: Path to the GRMHD dump file (HDF5).
- `MBH`: Black hole mass, in solar masses.
- `th_beg`: Polar angle cutoff for the radiating region, near the poles.
- `Rlow`, `Rhigh`, `beta_crit`: Mixed electron-temperature-model parameters.
- `sigma_cut`, `sigma_cut_high`: Magnetization cutoffs for the emitting
  region.
- `slow_light`: Whether slow-light (time-dependent) interpolation is
  enabled.
- `M_unit`: Mass unit, in g. Sets the overall density/temperature scale for
  the dump, and therefore the entire radiative transfer calculation. `Constants.M_UNIT_MAD` (6.e24) and
  `Constants.M_UNIT_SANE` (3.e26, this keyword's default) cover the two
  common GRMHD magnetic topologies: MAD and SANE for M87*; Ignored if the dump itself carries a
  `RADIATION` block with its own `M_unit` (read from the file instead).

# Returns
- The populated [`IharmParams`](@ref).
"""
function read_header(filename::String, MBH; th_beg=1.74e-2, Rlow=1.0, Rhigh=20.0, beta_crit=1.0, sigma_cut=1.0, sigma_cut_high=-1.0, slow_light=false, M_unit=3.e26)
    println("Initializing grid from: $filename")
    params = IharmParamsBuilder(MBH; th_beg, Rlow, Rhigh, beta_crit, sigma_cut, sigma_cut_high, slow_light, M_unit)
    cstopx_2 = 0.0

    h5open(filename, "r") do file
        header = file["header"]
        if haskey(header, "has_electrons")
            params.ELECTRONS = read(header, "has_electrons")
        else
            params.ELECTRONS = 0
        end

        if haskey(header, "has_radiation")
            params.RADIATION = read(header, "has_radiation")
        else
            params.RADIATION = 0
        end

        if haskey(header, "has_derefine_poles")
            error("Dump includes flag 'has_derefine_poles' and is therefore non-standard and not well-defined")
        end

        if haskey(header, "weights")
            weights = header["weights"]
            params.mu_i = read(weights, "mu_i")
            params.mu_e = read(weights, "mu_e")
            params.mu_tot = read(weights, "mu_tot")
            @printf(stderr, "Loaded molecular weights (mu_i, mu_e, mu_tot): %g %g %g\n", params.mu_i, params.mu_e, params.mu_tot)
            params.Ne_factor = 1.0 / params.mu_e
            params.ELECTRONS = ELECTRONS_TFLUID
        end

        metric_name_str = read(header, "metric")

        if startswith(metric_name_str, "MKS3")
            params.metric = Metrics.METRIC_MKS3
            cstopx_2 = 1.0
        elseif startswith(metric_name_str, "MKS")
            params.metric = Metrics.METRIC_MKS
            cstopx_2 = 1.0
        elseif startswith(metric_name_str, "BHAC_MKS")
            params.metric = Metrics.METRIC_BHACMKS
            cstopx_2 = π
        elseif startswith(metric_name_str, "MMKS") || startswith(metric_name_str, "FMKS")
            params.metric = Metrics.METRIC_FMKS
            cstopx_2 = 1.0
        elseif startswith(metric_name_str, "EKS")
            params.metric = Metrics.METRIC_EKS
            cstopx_2 = π
        else
            error("File is in unknown metric $metric_name_str. Cannot continue.")
        end

        params.N1 = read(header, "n1")
        params.N2 = read(header, "n2")
        params.N3 = read(header, "n3")
        params.gam = read(header, "gam")

        if haskey(header, "gam_e")
            @printf(stderr, "custom electron model loaded from dump file...\n")
            params.game = read(header, "gam_e")
            params.gamp = read(header, "gam_p")
        end
        params.Te_unit = params.Thetae_unit

        if !USE_FIXED_TPTE && !USE_MIXED_TPTE
            if params.ELECTRONS != 1
                error("! no electron temperature model specified! Cannot continue")
            end
            params.ELECTRONS = 1
            params.Thetae_unit = Constants.MP / Constants.ME
        elseif params.ELECTRONS == ELECTRONS_TFLUID
            @printf(stderr, "Using Ressler/Athena electrons with mixed tp_over_te and\n")
            @printf(stderr, "Rlow = %g, Rhigh = %g, and beta_crit = %g\n", params.Rlow, params.Rhigh, params.beta_crit)
        elseif USE_FIXED_TPTE && !USE_MIXED_TPTE
            params.ELECTRONS = 0
            params.Thetae_unit = 2.0 / 3.0 * Constants.MP / Constants.ME / (2.0 + params.tp_over_te)
        elseif USE_MIXED_TPTE && !USE_FIXED_TPTE
            params.ELECTRONS = 2
        else
            error("Unknown electron model $(params.ELECTRONS)! Cannot continue.")
        end

        params.Te_unit = params.Thetae_unit

        if params.RADIATION == 1
            @printf(stderr, "custom radiation field tracking information loaded...\n")
            units = header["units"]
            params.M_unit = read(units, "M_unit")
            params.T_unit = read(units, "T_unit")
            params.L_unit = read(units, "L_unit")
            params.Thetae_unit = read(units, "Thetae_unit")
            params.MBH = read(units, "Mbh")
            params.tp_over_te = read(units, "tp_over_te")
        end

        geom = header["geom"]
        params.startx[2] = read(geom, "startx1")
        params.startx[3] = read(geom, "startx2")
        params.startx[4] = read(geom, "startx3")
        params.dx[2] = read(geom, "dx1")
        params.dx[3] = read(geom, "dx2")
        params.dx[4] = read(geom, "dx3")

        local mks_group
        if params.metric == Metrics.METRIC_MKS
            mks_group = geom["mks"]
            @printf(stderr, "Using Modified Kerr-Schild coordinates MKS\n")
        elseif params.metric == Metrics.METRIC_BHACMKS
            mks_group = geom["bhac_mks"]
            @printf(stderr, "Using BHAC-style Modified Kerr-Schild coordinates BHAC_MKS\n")
        elseif params.metric == Metrics.METRIC_FMKS
            mks_group = geom["mmks"]
            @printf(stderr, "Using Funky Modified Kerr-Schild coordinates FMKS\n")
        elseif params.metric == Metrics.METRIC_MKS3
            mks_group = geom["mks3"]
            @printf(stderr, "Using logarithmic KS coordinates internally\n")
        elseif params.metric == Metrics.METRIC_EKS
            mks_group = geom["eks"]
            @printf(stderr, "Using Kerr-Schild coordinates with exponential radial coordinate\n")
        end

        if params.metric == Metrics.METRIC_MKS3
            params.a = read(mks_group, "a")
            params.mks3R0 = read(mks_group, "R0")
            params.mks3H0 = read(mks_group, "H0")
            params.mks3MY1 = read(mks_group, "MY1")
            params.mks3MY2 = read(mks_group, "MY2")
            params.mks3MP0 = read(mks_group, "MP0")
            params.Rout = 100.0
        elseif params.metric == Metrics.METRIC_EKS
            params.a = read(mks_group, "a")
            params.Rin = read(mks_group, "r_in")
            params.Rout = read(mks_group, "r_out")
            @printf(stderr, "eKS parameters a: %f Rin: %f Rout: %f\n", params.a, params.Rin, params.Rout)
        else
            params.a = read(mks_group, "a")
            params.hslope = read(mks_group, "hslope")

            if haskey(mks_group, "Rin")
                params.Rin = read(mks_group, "Rin")
                params.Rout = read(mks_group, "Rout")
            else
                params.Rin = read(mks_group, "r_in")
                params.Rout = read(mks_group, "r_out")
            end
            @printf(stderr, "MKS parameters a: %f hslope: %f Rin: %f Rout: %f\n", params.a, params.hslope, params.Rin, params.Rout)

            if params.metric == Metrics.METRIC_FMKS
                params.poly_xt = read(mks_group, "poly_xt")
                params.poly_alpha = read(mks_group, "poly_alpha")
                params.mks_smooth = read(mks_group, "mks_smooth")
            end
        end
    end

    return IharmParams(params, cstopx_2)
end


"""
    load_data(filename, Rhigh, model; advance_path! = nothing)

Read an iharm3d dump and build its IharmData snapshot. A uniform dump is stored as a mesh
with a single block covering the whole grid, with one layer of ghost cells around it.
"""
function load_data(filename::String, Rhigh, model::IharmParams; advance_path!::Union{Nothing,Function}=nothing)
    println("Loading data from '$filename' into 'Iharm' module...")
    !isfile(filename) && error("File not found: $filename")

    mesh = Grid.uniform_mesh(model)
    ng = mesh.ng

    t, fields = h5open(filename, "r") do file
        t = read(file, "t")
        haskey(file, "prims") || error("Dataset 'prims' not found in the HDF5 file.")
        prims = read(file["prims"])
        size(prims)[2:4] == (model.N3, model.N2, model.N1) ||
            error("Data dimensions $(reverse(size(prims)[2:4])) do not match expected grid size ($(model.N1), $(model.N2), $(model.N3))")

        # One block with a ghost layer: the dump goes in the inside, build_data fills the ghosts
        fields = [zeros(model.N1 + 2ng[1], model.N2 + 2ng[2], model.N3 + 2ng[3], 1) for _ in VALID_PRIMS]
        Threads.@threads for i in eachindex(VALID_PRIMS)
            inside = view(fields[i], ng[1]+1:ng[1]+model.N1, ng[2]+1:ng[2]+model.N2, ng[3]+1:ng[3]+model.N3, 1)
            permutedims!(inside, view(prims, i, :, :, :), (3, 2, 1))
        end
        t, fields
    end

    data = build_data(t, fields..., mesh, Rhigh, model)
    println("All primitives successfully loaded. Dimensions: ", size(data.RHO))

    advance_path! === nothing || advance_path!()
    return data
end

"""
    compute_accretion_diagnostics(model, data)

Mass accretion rate (Mdot), Eddington accretion rate (MdotEdd) and advected energy flux
(Ladv) of a snapshot, as in ipole: the radial mass and energy fluxes, integrated over theta
and phi and averaged over the 21 innermost radial zones of the grid.

With mesh refinement those zones may hold smaller cells; each cell then counts in
proportion to its size, so the result is the same average over the same shell.

# Returns
- `(Mdot, MdotEdd, Ladv)`, all in cgs.
"""
function compute_accretion_diagnostics(model::IharmParams, data::IharmData)
    mesh = data.mesh
    nb, ng = mesh.nb, mesh.ng

    # The shell is the n_shell innermost zones of the root grid in X1
    n_shell = min(21, model.N1)
    xshell = mesh.xmin[1] + n_shell * model.dx[2]

    dMact = 0.0
    Ladv = 0.0
    for mb in axes(data.RHO, 4)
        # cells of this block inside the shell (0 if the block is further out)
        nshell_b = clamp(round(Int, (xshell - mesh.bstart[1, mb]) / mesh.bdx[1, mb]), 0, nb[1])
        nshell_b == 0 && continue
        # weight of one cell: its volume in native coordinates over the width of the shell
        w = mesh.bdx[1, mb] * mesh.bdx[2, mb] * mesh.bdx[3, mb] / (n_shell * model.dx[2])

        for jj in 1:nb[2], ii in 1:nshell_b
            i = ii + ng[1]; j = jj + ng[2]
            # the metric depends only on X1 and X2
            X = SVector(0.0,
                mesh.bstart[1, mb] + (ii - 0.5) * mesh.bdx[1, mb],
                mesh.bstart[2, mb] + (jj - 0.5) * mesh.bdx[2, mb],
                mesh.bstart[3, mb] + 0.5 * mesh.bdx[3, mb])
            gcov = Metrics.gcov_func(X, model.a, model)
            gcon = Metrics.gcon_func(gcov)
            g = Metrics.gdet_func(gcov)

            for kk in 1:nb[3]
                k = kk + ng[3]    
                U = SVector(data.U1[i, j, k, mb], data.U2[i, j, k, mb], data.U3[i, j, k, mb])
                B = SVector(data.B1[i, j, k, mb], data.B2[i, j, k, mb], data.B3[i, j, k, mb])
                ucon, ucov, _, _ = fluid_fourv(Float64, U, B, gcov, gcon)

                # ucon[2]/ucov[1] are the radial-contravariant and time-covariant components,
                # matching ipole's ucon[1]/ucov[0]
                dMact += w * g * data.RHO[i, j, k, mb] * ucon[2]
                Ladv += w * g * data.UU[i, j, k, mb] * ucon[2] * ucov[1]
            end
        end
    end

    Mdot = -dMact * model.M_unit / model.T_unit
    MdotEdd = 4π * Constants.GNEWT * (model.MBH * Constants.MSUN) * Constants.MP /
            Constants.CL / 0.1 / Constants.SIGMA_THOMPSON

    return Mdot, MdotEdd, Ladv
end

 """
    init_physical_quantities(data, n, model, Rhigh)

Compute, for every cell of snapshot data[n], the quantities the radiative transfer needs:
electron density, electron temperature, magnetization (sigma), plasma beta, and the
derivative of the electron temperature with respect to Rhigh. They come from the density,
internal energy and magnetic field strength already stored in the snapshot, using the
Rlow/Rhigh electron temperature model.

Each cell is computed on its own, so this works for any grid layout and also fills the
ghost cells.
"""
function init_physical_quantities(data, n::Int64, model::IharmParams, Rhigh::T2) where {T2}
    rho_factor = model.RHO_unit / (Constants.MP + Constants.ME) * model.Ne_factor
    gam_minus_1 = model.gam - 1.0
    beta_crit_sq = model.beta_crit * model.beta_crit
    θe_factor = (Constants.MP / Constants.ME) * (model.game - 1.0) * (model.gamp - 1.0)
    game_minus_1 = model.game - 1.0
    gamp_minus_1 = model.gamp - 1.0
    B_unit_inv = 1.0 / model.B_unit

    ne_arr = data[n].ne
    b_arr = data[n].b
    θe_arr = data[n].θe
    sigma_arr = data[n].sigma
    RHO_arr = data[n].RHO
    UU_arr = data[n].UU

    # Purely per cell: loop over every cell of every block, ghosts included.
    # eachindex over all the arrays also checks that they have the same shape.
    @inbounds Threads.@threads for c in eachindex(ne_arr, b_arr, θe_arr, sigma_arr, RHO_arr, UU_arr)
        rho_c = RHO_arr[c]
        uu_c = UU_arr[c]
        b_c = b_arr[c]  # already correctly B_unit-scaled by the caller

        ne_arr[c] = rho_c * rho_factor

        bsq_normalized = b_c * B_unit_inv
        bsq = bsq_normalized * bsq_normalized

        sigma_m = bsq / rho_c
        beta_m = uu_c * gam_minus_1 / (0.5 * bsq)

        betasq = beta_m * beta_m / beta_crit_sq
        betasq_plus_1_inv = 1.0 / (1.0 + betasq)
        trat = Rhigh * betasq * betasq_plus_1_inv + model.Rlow * betasq_plus_1_inv
        θe_unit = θe_factor / (game_minus_1 * trat + gamp_minus_1)
        θe_val = θe_unit * uu_c / rho_c

        θe_arr[c] = MaxwellJuettnerPol.primal(θe_val) > 1.0e-3 ? θe_val : 1.0e-3
        sigma_arr[c] = MaxwellJuettnerPol.primal(sigma_m) > Constants.SMALL ? sigma_m : Constants.SMALL
    end
end

"""
    set_tinterp_ns(X, model, data)

Select the two GRMHD snapshots used for time interpolation, along with
the per-position time-interpolation factor. Returns `(dataA, dataB,
tinterp)` — the snapshots themselves, rather than their indices into
`data`.

On the CPU, `data` is a `Vector` and this dispatches to the generic
method below, branching on `model.slow_light` at runtime exactly as
before. On the GPU, `data` is a `Tuple` of one (non-slow-light) or three
(slow-light) snapshots; indexing a `Tuple` with a runtime-computed index
isn't GPU-compilable (even inside a branch that's never actually taken,
since `model.slow_light` is a runtime field, not a compile-time
constant, so the compiler can't prove that branch dead and skip
compiling it) — so the `Tuple`-specific methods below select on the
tuple's length instead, which is resolved at compile time.
"""
function set_tinterp_ns(X, model::IharmParams, data)
    if model.slow_light
        nA, nB = X[1] < data[2].t ? (1, 2) : (2, 3)
        dataA, dataB = data[nA], data[nB]
        tinterp = clamp(1.0 - (X[1] - dataA.t) / (dataB.t - dataA.t), 0.0, 1.0)
        return dataA, dataB, tinterp
    else
        return data[1], data[1], 0.0
    end
end

"""
    set_tinterp_ns(X, model, data::Tuple{T}) where {T}

GPU, non-slow-light variant of [`set_tinterp_ns`](@ref): `data` is a
1-tuple, so there's nothing to interpolate between.
"""
@inline set_tinterp_ns(X, model::IharmParams, data::Tuple{T}) where {T} = (data[1], data[1], 0.0)

"""
    set_tinterp_ns(X, model, data::NTuple{3,T}) where {T}

GPU, slow-light variant of [`set_tinterp_ns`](@ref): `data` is a 3-tuple
of snapshots, and the pair to interpolate between is selected by
comparing `X[1]` against the middle snapshot's time (resolved at compile
time via the tuple's length, unlike the `Vector` method's runtime
`model.slow_light` branch).
"""
@inline function set_tinterp_ns(X, model::IharmParams, data::NTuple{3,T}) where {T}
    if X[1] < data[2].t
        dataA, dataB = data[1], data[2]
    else
        dataA, dataB = data[2], data[3]
    end
    tinterp = 1.0 - (X[1] - dataA.t) / (dataB.t - dataA.t)
    return dataA, dataB, tinterp
end

"""
    get_model_sigma(zone, X, model, data)

Interpolate the magnetization `sigma` (ratio of magnetic to matter energy
density) at position `X`.

# Arguments
- `zone`: A `Grid.ZoneLoc` struct containing the zone indices, interpolation weights, and domain flag.
- `X`: Position four-vector in internal coordinates.
- `model`: Iharm model parameters.
- `data`: GRMHD snapshot(s).

# Returns
- The interpolated magnetization, or `0` if `X` is outside the grid.
"""
@inline function get_model_sigma(zone::Grid.ZoneLoc, X, model::IharmParams, data)
    T = promote_type(typeof(zone.del2), eltype(data[1].sigma))
    !zone.in_domain && return zero(T)
    dataA, dataB, tfac = set_tinterp_ns(X, model, data)
    return Grid.interp_scalar_time(zone, dataA.sigma, dataB.sigma, tfac, model.slow_light)
end

"""
    get_sigma_smoothfac(sigma, model)

Compute the smooth cutoff factor that tapers the electron density to zero
as the magnetization `sigma` approaches `model.sigma_cut`/
`model.sigma_cut_high`, avoiding a hard discontinuity at the cutoff.

# Arguments
- `sigma`: Magnetization at the position of interest.
- `model`: Iharm model parameters (`model.sigma_cut`,
  `model.sigma_cut_high`).

# Returns
- `1` well below the cutoff, `0` well above it, and a smooth cosine
  taper in between.
"""
function get_sigma_smoothfac(sigma, model::IharmParams)
    T = typeof(sigma)
    sigma_above = model.sigma_cut
    if model.sigma_cut_high > 0.0
        sigma_above = model.sigma_cut_high
    end
    if sigma < model.sigma_cut
        return one(T)
    end
    if sigma >= sigma_above
        return zero(T)
    end
    dsig = sigma_above - model.sigma_cut
    return cos(π / 2.0 / dsig * (sigma - model.sigma_cut))
end

"""
    get_model_ne(zone, X, model, data)

Interpolate the electron number density at position `X`, applying the
geodesic magnetization cutoff (via [`get_model_sigma`](@ref)/
[`get_sigma_smoothfac`](@ref)) if `USE_GEODESIC_SIGMACUT` is enabled.

# Arguments
- `zone`: A `Grid.ZoneLoc` struct containing the zone indices, interpolation weights, and domain flag.
- `X`: Position four-vector in internal coordinates.
- `model`: Iharm model parameters.
- `data`: GRMHD snapshot(s).

# Returns
- The electron number density, or `0` if `X` is outside the grid or above
  the magnetization cutoff.
"""
@inline function get_model_ne(zone::Grid.ZoneLoc, X, model::IharmParams, data)
    T = promote_type(typeof(zone.del2), eltype(data[1].ne))
    !zone.in_domain && return zero(T)
    sigma_smoothfac = one(T)
    if USE_GEODESIC_SIGMACUT
        sigma = get_model_sigma(zone, X, model, data)
        MaxwellJuettnerPol.primal(sigma) > model.sigma_cut && return zero(T)
        sigma_smoothfac = get_sigma_smoothfac(sigma, model)
    end
    dataA, dataB, tfac = set_tinterp_ns(X, model, data)
    return Grid.interp_scalar_time(zone, dataA.ne, dataB.ne, tfac, model.slow_light) * sigma_smoothfac
end

"""
    get_model_thetae(zone, X, model, data)

Interpolate the dimensionless electron temperature at position `X`, using the zone location `zone`.

# Arguments
- `zone`: A `Grid.ZoneLoc` struct containing the zone indices, interpolation weights, and domain flag.
- `X`: Position four-vector in internal coordinates.
- `model`: Iharm model parameters.
- `data`: GRMHD snapshot(s).

# Returns
- The dimensionless electron temperature, or `0` if `X` is outside the
  grid or if the zone is not in the domain.
"""
@inline function get_model_thetae(zone::Grid.ZoneLoc, X, model::IharmParams, data)
    T = promote_type(typeof(zone.del2), eltype(data[1].θe))
    !zone.in_domain && return zero(T)
    dataA, dataB, tfac = set_tinterp_ns(X, model, data)
    return Grid.interp_scalar_time(zone, dataA.θe, dataB.θe, tfac, model.slow_light)
end

"""
    get_model_b(zone, X, model, data)

Interpolate the magnetic field strength at position `X`.

# Arguments
- `zone`: A `Grid.ZoneLoc` struct containing the zone indices, interpolation weights, and domain flag.
- `X`: Position four-vector in internal coordinates.
- `model`: Iharm model parameters.
- `data`: GRMHD snapshot(s).

# Returns
- The magnetic field strength, or `0` if `X` is outside the grid.
"""
@inline function get_model_b(zone::Grid.ZoneLoc, X, model::IharmParams, data)
    T = promote_type(typeof(zone.del2), eltype(data[1].b))
    !zone.in_domain && return zero(T)
    dataA, dataB, tfac = set_tinterp_ns(X, model, data)
    return Grid.interp_scalar_time(zone, dataA.b, dataB.b, tfac, model.slow_light)
end


"""
    fluid_fourv(T, V, B, gcov, gcon)

Fluid four-velocity and magnetic field four-vector from the GRMHD primitives at one
point: `V` holds the velocity primitives (U1, U2, U3), `B` the field primitives
(B1, B2, B3), and `gcov`/`gcon` are the metric there. The vectors are built with
element type `T`. Used by the snapshot loader, the accretion diagnostics and the
per-step [`get_model_fourv`](@ref).

# Returns
- A tuple `(Ucon, Ucov, Bcon, Bcov)`.
"""
@inline function fluid_fourv(::Type{T}, V, B, gcov, gcon) where {T}
    VdotV = zero(T)
    for l in 1:3
        for m in 1:3
            VdotV += gcov[l+1, m+1] * V[l] * V[m]
        end
    end
    Vfac = sqrt(-one(T) / gcon[1, 1] * (one(T) + abs(VdotV)))
    Ucon = SVector{4,T}(-Vfac * gcon[1, 1], V[1] - Vfac * gcon[1, 2], V[2] - Vfac * gcon[1, 3], V[3] - Vfac * gcon[1, 4])
    Ucov = Coordinates.flip_index(Ucon, gcov)

    udotB = Ucov[2] * B[1] + Ucov[3] * B[2] + Ucov[4] * B[3]
    Bcon = SVector{4,T}(udotB,
        (B[1] + Ucon[2] * udotB) / Ucon[1],
        (B[2] + Ucon[3] * udotB) / Ucon[1],
        (B[3] + Ucon[4] * udotB) / Ucon[1])
    Bcov = Coordinates.flip_index(Bcon, gcov)
    return Ucon, Ucov, Bcon, Bcov
end

"""
    get_model_fourv(zone, X, Kcon, bhspin, model, data)

Compute the fluid 4-velocity and magnetic field 4-vector at `X`,
interpolated from the GRMHD snapshot(s) in `data`.

`bhspin` is a separate argument (rather than `model.a`) for consistency
with the rest of the differentiable call path.

# Arguments
- `zone`: A `Grid.ZoneLoc` struct containing the zone indices, interpolation weights, and domain flag.
- `X`: Position four-vector in internal coordinates.
- `Kcon`: Contravariant photon 4-momentum (unused; kept for interface
  symmetry with the other models' four-velocity functions).
- `bhspin`: Dimensionless black hole spin parameter.
- `model`: Iharm model parameters.
- `data`: GRMHD snapshot(s).

# Returns
- A tuple `(Ucon, Ucov, Bcon, Bcov)`.
"""
@inline function get_model_fourv(zone::Grid.ZoneLoc, X, Kcon, bhspin, model::IharmParams, data)
    elT = promote_type(eltype(X), eltype(data[1].θe))
    gcov = Metrics.gcov_func(X, bhspin, model)
    gcon = Metrics.gcon_func(gcov)

    if !zone.in_domain
        Ucov1 = -one(elT) / sqrt(-gcon[1, 1])
        Ucon1 = Ucov1 * gcon[1, 1]; Ucon2 = Ucov1 * gcon[2, 1]
        Ucon3 = Ucov1 * gcon[3, 1]; Ucon4 = Ucov1 * gcon[4, 1]
        Ucon = SVector{4,elT}(Ucon1, Ucon2, Ucon3, Ucon4)
        Ucov = SVector{4,elT}(Ucov1, zero(elT), zero(elT), zero(elT))
        zero_vec = SVector{4,elT}(0.0, 0.0, 0.0, 0.0)
        return Ucon, Ucov, zero_vec, zero_vec
    end

    dataA, dataB, tfac = set_tinterp_ns(X, model, data)
    V = SVector(Grid.interp_scalar_time(zone, dataA.U1, dataB.U1, tfac, model.slow_light),
                Grid.interp_scalar_time(zone, dataA.U2, dataB.U2, tfac, model.slow_light),
                Grid.interp_scalar_time(zone, dataA.U3, dataB.U3, tfac, model.slow_light))
    B = SVector(Grid.interp_scalar_time(zone, dataA.B1, dataB.B1, tfac, model.slow_light),
                Grid.interp_scalar_time(zone, dataA.B2, dataB.B2, tfac, model.slow_light),
                Grid.interp_scalar_time(zone, dataA.B3, dataB.B3, tfac, model.slow_light))
    return fluid_fourv(elT, V, B, gcov, gcon)
end

"""
      Radiation.get_jk(X, Kcon, freq, bhspin, model::IharmParams, data)

  Emissivity and absorption coefficient (invariant, Stokes I) of the GRMHD model at
  position `X`. Zero outside the grid, above the magnetization cutoff, or when the ray
  runs exactly along the magnetic field.

  # Arguments
  - `X`: Position four-vector in internal coordinates.
  - `Kcon`: Contravariant photon 4-momentum (already carries the frequency).
  - `freq`: Observing frequency; unused by this model.
  - `bhspin`: Dimensionless black hole spin parameter.
  - `model`: Iharm model parameters.
  - `data`: GRMHD snapshot(s).

  # Returns
  - A tuple `(j, k)`.
  """
  function Radiation.get_jk(X, Kcon, freq, bhspin, model::IharmParams, data)
    zone = Grid.locate(X, model, data[1].mesh)
    Ne = get_model_ne(zone, X, model, data)
    z_base = zero(typeof(Ne))
    if MaxwellJuettnerPol.primal(Ne) == 0.0
        return (z_base, z_base)
    end

    elT = promote_type(eltype(X), typeof(bhspin))

    Ucon, Ucov, Bcon, Bcov = get_model_fourv(zone, X, Kcon, bhspin, model, data)
    nu = Radiation.get_fluid_nu(Kcon, Ucov)
    θ = Radiation.get_bk_angle(Kcon, Ucov, Bcon, Bcov)
    b = get_model_b(zone, X, model, data)

    θe = get_model_thetae(zone, X, model, data)
    if MaxwellJuettnerPol.primal(θ) <= zero(elT) || MaxwellJuettnerPol.primal(θ) >= elT(π)
        return (z_base, z_base)
    end
    j = MaxwellJuettner.maxwell_juettner_leung_i(Ne, nu, θe, b, θ) / (nu * nu)

    Bnuinv = Radiation.bnu_inv(nu, θe)
    k = (Bnuinv > 0) ? j / Bnuinv : zero(typeof(j))
    return (j, k)
end

"""
    Radiation.radiating_region(X, model::IharmParams, Rh)

Check whether `X` lies within the GRMHD grid's radiating region (within
the grid's radial bounds and away from the polar axis).
"""
function Radiation.radiating_region(X, model::IharmParams, Rh)
    r, th = Coordinates.bl_coord(X, model)
    return (r > (model.rmin_geo) && r < model.rmax_geo && th > model.th_beg && th < (π - model.th_beg))
end

"""
    Coordinates.bl_coord(X, model::IharmParams, R0=0.0)

Compute the Boyer-Lindquist coordinates `(r, th)` from the internal
coordinates `X`, using the FMKS/MKS polar mapping (`model.hslope`,
and, for FMKS, `model.poly_xt`/`model.poly_alpha`/`model.poly_norm`/
`model.mks_smooth`).
"""
function Coordinates.bl_coord(X, model::IharmParams, R0::Float64=0.0)
    r = exp(X[2]) + R0

    if model.metric == Metrics.METRIC_FMKS
        thG = π * X[3] + ((1.0 - model.hslope) / 2.0) * sin(2.0 * π * X[3])
        y = 2 * X[3] - 1.0
        thJ = model.poly_norm * y * (1.0 + ((y / model.poly_xt)^model.poly_alpha) / (model.poly_alpha + 1.0)) + 0.5 * π
        th = thG + exp(model.mks_smooth * (model.startx[2] - X[2])) * (thJ - thG)
    else
        th = π * X[3] + ((1.0 - model.hslope) / 2.0) * sin(2.0 * π * X[3])
    end
    return r, th
end


"""
    Coordinates.set_ks_jacobian(X, model::IharmParams)

Compute the Jacobian matrix `dx/dX`, using the FMKS/MKS polar mapping.
"""
function Coordinates.set_ks_jacobian(X, model::IharmParams)
    T = eltype(X)

    m22 = exp(X[2])
    m32 = zero(T)

    if model.metric == Metrics.METRIC_MKS
        m33 = T(π) + (1 - model.hslope) * T(π) * cos(2 * T(π) * X[3])
    else
        m32 = -exp(model.mks_smooth * (model.startx[2] - X[2])) * model.mks_smooth * (π / 2 - π * X[3] + model.poly_norm * (2 * X[3] - 1) * (1 + ((-1 + 2 * X[3]) / model.poly_xt)^model.poly_alpha / (1 + model.poly_alpha)) - 0.5 * (1 - model.hslope) * sin(2 * π * X[3]))
        m33 = π + (1 - model.hslope) * π * cos(2 * π * X[3]) + exp(model.mks_smooth * (model.startx[2] - X[2])) * (-π + 2 * model.poly_norm * (1 + ((2 * X[3] - 1) / model.poly_xt)^model.poly_alpha / (model.poly_alpha + 1)) + (2 * model.poly_alpha * model.poly_norm * (2 * X[3] - 1) * ((2 * X[3] - 1) / model.poly_xt)^(model.poly_alpha - 1)) / ((1 + model.poly_alpha) * model.poly_xt) - (1 - model.hslope) * π * cos(2 * π * X[3]))
    end

    if m33 <= 0.0
        m33 = T(1.0e-10)
    end

    return SMatrix{4,4,T}(
        one(T), zero(T), zero(T), zero(T),
        zero(T), m22, m32, zero(T),
        zero(T), zero(T), m33, zero(T),
        zero(T), zero(T), zero(T), one(T)
    )
end

"""
    Camera.camera_position(cam_dist, cam_theta_angle, cam_phi_angle, bhspin, model::IharmParams)

Compute the camera position in native coordinates, numerically inverting
the FMKS/MKS polar mapping via [`Camera.root_find_std`](@ref) (unlike the
direct formula used by `Analytic`/`ThinDisk`).
"""
function Camera.camera_position(cam_dist, cam_theta_angle, cam_phi_angle, bhspin, model::IharmParams)
    T = promote_type(typeof(cam_dist), typeof(cam_theta_angle), typeof(cam_phi_angle), typeof(bhspin))
    x = @SVector [zero(T), T(cam_dist), T(cam_theta_angle) / T(180) * T(π), T(cam_phi_angle) / T(180) * T(π)]
    x3_val = Camera.root_find_std(x, model)
    return @SVector [
        zero(T),
        log(cam_dist),
        x3_val,
        (cam_phi_angle / 180) * π
    ]
end


# This is the same as load_grmhd_context, but we don't need to load the data from a file,
# the file is already loaded in memory
function grmhd_context(params0::IharmParams, data0::IharmData)
    b_normalized = data0.b ./ params0.B_unit
    return (; N1=params0.N1, N2=params0.N2, N3=params0.N3,
              dx=params0.dx, startx=params0.startx, stopx=params0.stopx,
              cstartx=params0.cstartx, cstopx=params0.cstopx,
              metric=params0.metric, a=params0.a, hslope=params0.hslope,
              Rin=params0.Rin, Rout=params0.Rout, rmin_geo=params0.rmin_geo, rmax_geo=params0.rmax_geo,
              poly_xt=params0.poly_xt, poly_alpha=params0.poly_alpha, mks_smooth=params0.mks_smooth,
              poly_norm=params0.poly_norm, mks3R0=params0.mks3R0, mks3H0=params0.mks3H0,
              mks3MY1=params0.mks3MY1, mks3MY2=params0.mks3MY2, mks3MP0=params0.mks3MP0,
              gam=params0.gam, game=params0.game, gamp=params0.gamp,
              mu_i=params0.mu_i, mu_e=params0.mu_e, mu_tot=params0.mu_tot, Ne_factor=params0.Ne_factor,
              ELECTRONS=params0.ELECTRONS, RADIATION=params0.RADIATION, tp_over_te=params0.tp_over_te,
              slow_light=params0.slow_light, t=data0.t,
              RHO=data0.RHO, UU=data0.UU, U1=data0.U1, U2=data0.U2, U3=data0.U3,
              B1=data0.B1, B2=data0.B2, B3=data0.B3, b_normalized=b_normalized, mesh=data0.mesh)
end



const GRADIENT_PARAM_NAMES = (:MBH, :Rhigh, :Rlow, :beta_crit, :th_beg, :sigma_cut, :sigma_cut_high,
                               :M_unit, :ro, :th, :phi, :sourceD)

"""
    dual_problem(ctx; MBH, Rhigh, Rlow, beta_crit, th_beg, sigma_cut, sigma_cut_high,
                M_unit, ro, th, phi, sourceD, wrt=())

Build the dual-number copy of a GRMHD problem for forward-mode automatic
differentiation: every parameter named in `wrt` is seeded with its own partial,
the units and the model are rebuilt from the seeded values, and the derived
plasma arrays (`ne`, `θe`, `b`, `sigma`) are recomputed so that they carry the
partials too.

# Arguments
- `ctx`: A context from [`grmhd_context`](@ref).
- keyword arguments: values of the differentiable parameters (see
`GRADIENT_PARAM_NAMES`); `wrt` lists the ones to differentiate.

# Returns
- A named tuple `(model, data, ro, th, phi, sourceD)` of dual-typed objects,
with `data` a one-element vector as `Imaging.raytrace_image` expects.
"""
function dual_problem(ctx; MBH, Rhigh, Rlow, beta_crit, th_beg, sigma_cut, sigma_cut_high, M_unit,
        ro, th, phi, sourceD, wrt::NTuple{N,Symbol}=()) where N

    dualize(val, sym) = sym in wrt ? ForwardDiff.Dual{Nothing,Float64,N}(val, ForwardDiff.Partials(ntuple(i -> i == findfirst(==(sym), wrt) ? 1.0 : 0.0, N))) : val

    MBH_d, Rhigh_d, Rlow_d = dualize(MBH, :MBH), dualize(Rhigh, :Rhigh), dualize(Rlow, :Rlow)
    beta_crit_d, th_beg_d = dualize(beta_crit, :beta_crit), dualize(th_beg, :th_beg)
    sigma_cut_d, sigma_cut_high_d = dualize(sigma_cut, :sigma_cut), dualize(sigma_cut_high, :sigma_cut_high)
    M_unit_d = dualize(M_unit, :M_unit)
    ro_d, th_d, phi_d, sourceD_d = dualize(ro, :ro), dualize(th, :th), dualize(phi, :phi), dualize(sourceD, :sourceD)

    T = promote_type(typeof(MBH_d), typeof(Rhigh_d), typeof(Rlow_d), typeof(beta_crit_d), typeof(th_beg_d),
                    typeof(sigma_cut_d), typeof(sigma_cut_high_d), typeof(M_unit_d),
                    typeof(ro_d), typeof(th_d), typeof(phi_d), typeof(sourceD_d))

    L_unit_d = Constants.GNEWT * MBH_d * Constants.MSUN / Constants.CL^2
    T_unit_d = L_unit_d / Constants.CL
    RHO_unit_d = M_unit_d / L_unit_d^3
    U_unit_d = RHO_unit_d * Constants.CL^2
    B_unit_d = Constants.CL * sqrt(4π * RHO_unit_d)

    model_d = IharmParams{T}(ctx.metric, ctx.ELECTRONS, ctx.RADIATION,
        ctx.gam, ctx.game, ctx.gamp, (0.0), (0.0),
        ctx.mu_i, ctx.mu_e, ctx.mu_tot, ctx.Ne_factor,
        M_unit_d, T_unit_d, L_unit_d, MBH_d, ctx.tp_over_te,
        RHO_unit_d, U_unit_d, B_unit_d, ctx.a, ctx.hslope, ctx.Rin, ctx.Rout,
        ctx.poly_xt, ctx.poly_alpha, ctx.mks_smooth, ctx.poly_norm,
        ctx.mks3R0, ctx.mks3H0, ctx.mks3MY1, ctx.mks3MY2, ctx.mks3MP0,
        ctx.N1, ctx.N2, ctx.N3, ctx.dx, ctx.startx, ctx.stopx, ctx.cstartx, ctx.cstopx,
        ctx.rmin_geo, ctx.rmax_geo, th_beg_d, Rlow_d, Rhigh_d, beta_crit_d, sigma_cut_d,
        sigma_cut_high_d, ctx.slow_light)

    b_d = ctx.b_normalized .* T(B_unit_d)
    data_d = IharmData(ctx.t, ctx.RHO, ctx.UU, ctx.U1, ctx.U2, ctx.U3, ctx.B1, ctx.B2, ctx.B3, similar(ctx.RHO, T), b_d, similar(ctx.RHO, T), similar(ctx.RHO, T), ctx.mesh)

    init_physical_quantities([data_d], 1, model_d, Rhigh_d)

    return (model=model_d, data=[data_d], ro=ro_d, th=th_d, phi=phi_d, sourceD=sourceD_d)
end

"""
    calculate_gradients(ctx, freq, pixels_x, pixels_y, fovx_uas, maxnstep, xoff, yoff;
        MBH, Rhigh, Rlow, beta_crit, th_beg, sigma_cut, sigma_cut_high, M_unit,
        ro, th, phi, sourceD, wrt=(), pol=false)

The image (and, with `pol=true`, the polarized image) and their derivatives with
respect to the parameters in `wrt`, from one forward-mode pass of dual numbers
through the ordinary ray-tracing code.

# Arguments
- `ctx`: A context from [`grmhd_context`](@ref).
- `freq`: Observing frequency, in Hz.
- `pixels_x`, `pixels_y`: Image resolution.
- `fovx_uas`: Field of view, in microarcseconds.
- `maxnstep`: Maximum number of geodesic steps.
- `xoff`, `yoff`: Image plane offsets.
- keyword arguments: values of the differentiable parameters (see
`GRADIENT_PARAM_NAMES`); `wrt` lists the ones to differentiate.
- `pol`: also compute the polarized image and its derivatives.

# Returns
- `pol=false`: `Image` if `wrt` is empty, otherwise `(Image, grads)`.
- `pol=true`: `(Image, pol)` if `wrt` is empty, otherwise `(Image, pol, grads, grads_pol)`.

`Image` is `(nx, ny)`, `pol` is `(NIMG, nx, ny)` (I, Q, U, V, Faraday depth), and
`grads`/`grads_pol` are named tuples with one array of the same shape per parameter.
"""
 function calculate_gradients(ctx, freq, pixels_x, pixels_y, fovx_uas, maxnstep, xoff, yoff;
        MBH, Rhigh, Rlow, beta_crit, th_beg, sigma_cut, sigma_cut_high, M_unit,
        ro, th, phi, sourceD, wrt::NTuple{N,Symbol}=(), pol::Bool=false) where N

    p = dual_problem(ctx; MBH=MBH, Rhigh=Rhigh, Rlow=Rlow, beta_crit=beta_crit, th_beg=th_beg,
        sigma_cut=sigma_cut, sigma_cut_high=sigma_cut_high, M_unit=M_unit,
        ro=ro, th=th, phi=phi, sourceD=sourceD, wrt=wrt)

    Dxsize = p.sourceD / p.model.L_unit / Constants.MUAS_PER_RAD * fovx_uas
    fovx_d = Dxsize / p.ro
    fovy_d = Dxsize / p.ro
    Rh = 1.0 + sqrt(1.0 - ctx.a^2)

    # One array of derivatives per parameter in wrt.
    grads_of(A) = NamedTuple{wrt}(Tuple(ForwardDiff.partials.(A, i) for i in 1:N))

    if pol
        Image_dual, pol_dual = ImagingPol.raytrace_image_pol(p.model, p.data, p.ro, p.th, p.phi, freq,
            pixels_x, pixels_y, fovx_d, fovy_d, maxnstep, Rh, xoff, yoff)
        Image, pol_img = ForwardDiff.value.(Image_dual), ForwardDiff.value.(pol_dual)
        N == 0 && return Image, pol_img
        return Image, pol_img, grads_of(Image_dual), grads_of(pol_dual)
    end

    Image_dual = Imaging.raytrace_image(p.model, p.data, p.ro, p.th, p.phi, freq,
        pixels_x, pixels_y, fovx_d, fovy_d, maxnstep, Rh, xoff, yoff)
    Image = ForwardDiff.value.(Image_dual)
    N == 0 && return Image
    return Image, grads_of(Image_dual)
end

"""
      plasma_state(X, Kcon, bhspin, model, data)

  Local plasma quantities that set the synchrotron transfer coefficients at `X`:
  the zone lookup, the fluid four-velocity and field, and from them the electron
  density, fluid-frame frequency, field strength, electron temperature and
  field-wavevector angle, evaluated in the order ipole's `jar_calc_dist` uses.

  Unlike `Radiation.get_jk`, the four-velocity and field are computed before the
  density check, because the polarized step needs the plasma frame even where
  there is no plasma.

  # Arguments
  - `X`: Position four-vector in internal coordinates.
  - `Kcon`: Contravariant photon 4-momentum.
  - `bhspin`: Dimensionless black hole spin parameter.
  - `model`: Iharm model parameters.
  - `data`: GRMHD snapshot(s).

  # Returns
  - A named tuple `(Ne, nu, θe, B, θ, Ucon, Ucov, Bcon, Bcov)`. Where `Ne <= 0`
    (outside the grid or above the magnetization cut) the other scalars are
    not evaluated and are returned as zero.
"""
@inline function plasma_state(X, Kcon, bhspin, model::IharmParams, data)
    zone = Grid.locate(X, model, data[1].mesh)
    Ne = get_model_ne(zone, X, model, data)
    Ucon, Ucov, Bcon, Bcov = get_model_fourv(zone, X, Kcon, bhspin, model, data)

    # Common number type of everything returned, so both branches agree.
    T = promote_type(typeof(Ne), eltype(Ucov), eltype(Bcov), eltype(Kcon),
        eltype(data[1].b), eltype(data[1].θe), typeof(zone.del2))
    z = zero(T)

    if !(primal(Ne) > 0)
        return (Ne=z, nu=z, θe=z, B=z, θ=z, Ucon=Ucon, Ucov=Ucov, Bcon=Bcon, Bcov=Bcov)
    end

    nu = Radiation.get_fluid_nu(Kcon, Ucov)
    θ = Radiation.get_bk_angle(Kcon, Ucov, Bcon, Bcov)
    B = get_model_b(zone, X, model, data)
    θe = get_model_thetae(zone, X, model, data)

    return (Ne=T(Ne), nu=T(nu), θe=T(θe), B=T(B), θ=T(θ), Ucon=Ucon, Ucov=Ucov, Bcon=Bcon, Bcov=Bcov)
end

"""
    Polarization.get_pol_state(X, Kcon, freq, bhspin, model::IharmParams, data)

Polarized transfer coefficients and plasma frame of the GRMHD model at `X`:
thermal synchrotron coefficients (`Polarization.thermal_jar`) for the local
[`plasma_state`](@ref). `freq` is unused (the fluid-frame frequency follows
from `Kcon`) and is kept for symmetry with `Radiation.get_jk`.
"""
@inline function Polarization.get_pol_state(X, Kcon, freq, bhspin, model::IharmParams, data)
    s = plasma_state(X, Kcon, bhspin, model, data)
    coeffs = Polarization.thermal_jar(s.Ne, s.nu, s.θe, s.B, s.θ)
    return coeffs, s.Ucon, s.Bcon, s.Bcov
end

end
