"""
GRMHD simulation-data emission model (`ipole`'s "iharm" model), reading
dumps in `iharm3d`/`KHARMA` HDF5 format.
"""
module Iharm

using HDF5
using Printf
using StaticArrays
using ForwardDiff
using ..Constants
using ..AbstractModels
using ..Coordinates
using ..Metrics
using ..Camera
using ..Grid
using ..Radiation
using ..MaxwellJuettner

export IharmParams, IharmData, read_header, load_data

const VALID_PRIMS = ["RHO", "UU", "U1", "U2", "U3", "B1", "B2", "B3"]
const USE_GEODESIC_SIGMACUT = true
const ELECTRONS_TFLUID = 3
const USE_FIXED_TPTE = false
const USE_MIXED_TPTE = true

"""
A single GRMHD simulation snapshot: fluid primitives on the simulation
grid, plus the derived electron/magnetic-field quantities used by the
radiative transfer.
"""
struct IharmData
    t::Float64
    RHO::Array{Float64,3}
    UU::Array{Float64,3}
    U1::Array{Float64,3}
    U2::Array{Float64,3}
    U3::Array{Float64,3}
    B1::Array{Float64,3}
    B2::Array{Float64,3}
    B3::Array{Float64,3}
    ne::Array{Float64,3}
    b::Array{Float64,3}
    θe::Array{Float64,3}
    sigma::Array{Float64,3}
    beta::Array{Float64,3}
    dθedRhi::Array{Float64,3}
end

"""
Parameters for the [`Iharm`](@ref) GRMHD emission model.

Populated from a dump file's header via [`read_header`](@ref); most
fields mirror `ipole`'s C `Params`/`Grid` globals.
"""
mutable struct IharmParams <: AbstractModel
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

    M_unit::Float64
    T_unit::Float64
    L_unit::Float64
    MBH::Float64
    tp_over_te::Float64

    RHO_unit::Float64
    U_unit::Float64
    B_unit::Float64

    a::Float64
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

    th_beg::Float64
    trat_small::Float64
    beta_crit::Float64
    sigma_cut::Float64
    sigma_cut_high::Float64
    slow_light::Bool
end

"""
    IharmParams()

Default-constructed [`IharmParams`](@ref), used internally by
[`read_header`](@ref) before the dump file's header is parsed into it.
"""
function IharmParams()
    return IharmParams(0, 0, 0,
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
        1.74e-2, 1.0, 1.0, 1.0, -1.0, false)
end

"""
    read_header(filename, MBH; th_beg=1.74e-2, trat_small=1.0, beta_crit=1.0, sigma_cut=1.0, sigma_cut_high=-1.0, slow_light=false, M_unit=3.e26)

Read a GRMHD dump file's header, returning the populated
[`IharmParams`](@ref).

The electron-temperature-model parameters (`th_beg`, `trat_small`,
`beta_crit`, `sigma_cut`, `sigma_cut_high`) are run configuration, not
recorded in the dump file, so they're supplied here (with `ipole`'s usual
defaults). `L_unit`/`T_unit`/`RHO_unit`/`U_unit`/`B_unit` are derived from
the black hole mass `MBH` and `M_unit`.

# Arguments
- `filename`: Path to the GRMHD dump file (HDF5).
- `MBH`: Black hole mass, in solar masses.
- `th_beg`: Polar angle cutoff for the radiating region, near the poles.
- `trat_small`, `beta_crit`: Mixed electron-temperature-model parameters.
- `sigma_cut`, `sigma_cut_high`: Magnetization cutoffs for the emitting
  region.
- `slow_light`: Whether slow-light (time-dependent) interpolation is
  enabled.
- `M_unit`: Mass unit, in g.

# Returns
- The populated [`IharmParams`](@ref).
"""
function read_header(filename::String, MBH; th_beg=1.74e-2, trat_small=1.0, beta_crit=1.0, sigma_cut=1.0, sigma_cut_high=-1.0, slow_light=false, M_unit=3.e26)
    println("Initializing grid from: $filename")

    params = IharmParams()
    params.th_beg = th_beg
    params.trat_small = trat_small
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

        if startswith(metric_name_str, "MKS")
            params.metric = Metrics.METRIC_MKS
            cstopx_2 = 1.0
        elseif startswith(metric_name_str, "BHAC_MKS")
            params.metric = Metrics.METRIC_BHACMKS
            cstopx_2 = π
        elseif startswith(metric_name_str, "MMKS") || startswith(metric_name_str, "FMKS")
            params.metric = Metrics.METRIC_FMKS
            cstopx_2 = 1.0
        elseif startswith(metric_name_str, "MKS3")
            params.metric = Metrics.METRIC_MKS3
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
            @printf(stderr, "trat_small = %g, trat_large = %g, and beta_crit = %g\n", params.trat_small, NaN, params.beta_crit)
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

                params.poly_norm = 0.5 * π * 1.0 / (1.0 + 1.0 / (params.poly_alpha + 1.0) * 1.0 / (params.poly_xt^params.poly_alpha))

                @printf(stderr, "FMKS parameters poly_xt: %f poly_alpha: %f mks_smooth: %f poly_norm: %f\n",
                    params.poly_xt, params.poly_alpha, params.mks_smooth, params.poly_norm)
            end
        end
    end

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

    if params.metric != Metrics.METRIC_BHACMKS && params.metric != Metrics.METRIC_EKS
        params.cstopx[2] = log(params.Rout)
    end

    @printf(stderr, "Grid start (startx): %.15e, %.15e, %.15e stop (stopx): %.15e, %.15e, %.15e\n",
        params.startx[2], params.startx[3], params.startx[4], params.stopx[2], params.stopx[3], params.stopx[4])
    @printf(stderr, "grid dx: %.15e, %.15e, %.15e\n", params.dx[2], params.dx[3], params.dx[4])

    params.RHO_unit = params.M_unit / params.L_unit^3
    params.U_unit = params.RHO_unit * Constants.CL^2
    params.B_unit = Constants.CL * sqrt(4 * π * params.RHO_unit)

    return params
end

function _read_single_primitive(file_handle, prim_name::String)
    if "prims" in keys(file_handle)
        prims = read(file_handle["prims"])
        prim_idx = findfirst(isequal(uppercase(prim_name)), VALID_PRIMS)
        return prim_idx === nothing ? nothing : permutedims(prims[prim_idx, :, :, :], (3, 2, 1))
    else
        error("Dataset 'prims' not found in the HDF5 file.")
    end
end

"""
    load_data(filename, trat_large, model; advance_path!=nothing)

Load a GRMHD dump file's fluid primitives and compute the derived
electron/magnetic-field quantities used by the radiative transfer.

# Arguments
- `filename`: Path to the GRMHD dump file (HDF5).
- `trat_large`: Electron/ion temperature ratio at high magnetization
  (`Rhigh`).
- `model`: Iharm model parameters.
- `advance_path!`: Optional zero-argument callback invoked after a
  successful load (used by `Slowlight` to advance the dump sequence).

# Returns
- The loaded [`IharmData`](@ref).
"""
function load_data(filename::String, trat_large::Float64, model::IharmParams; advance_path!::Union{Nothing,Function}=nothing)
    println("Loading data from '$filename' into 'Iharm' module...")
    !isfile(filename) && error("File not found: $filename")

    rho = uu = u1 = u2 = u3 = b1 = b2 = b3 = nothing

    data_array = Vector{IharmData}(undef, 1)

    h5open(filename, "r") do file
        t = read(file, "t")
        Threads.@threads for prim_name in VALID_PRIMS
            data_3d = _read_single_primitive(file, prim_name)
            if data_3d !== nothing
                data_3d = Float64.(data_3d)

                if prim_name == "RHO"
                    rho = data_3d
                    if size(rho, 1) != model.N1 || size(rho, 2) != model.N2 || size(rho, 3) != model.N3
                        println("N1 = $(size(rho,1)), N2 = $(size(rho,2)), N3 = $(size(rho,3))")
                        error("Data dimensions do not match expected grid size N1,N2,N3")
                    end
                elseif prim_name == "UU"
                    uu = data_3d
                elseif prim_name == "U1"
                    u1 = data_3d
                elseif prim_name == "U2"
                    u2 = data_3d
                elseif prim_name == "U3"
                    u3 = data_3d
                elseif prim_name == "B1"
                    b1 = data_3d
                elseif prim_name == "B2"
                    b2 = data_3d
                elseif prim_name == "B3"
                    b3 = data_3d
                end
            end
        end
        data_array[1] = IharmData(t, rho, uu, u1, u2, u3, b1, b2, b3, zeros(size(rho)), zeros(size(rho)), zeros(size(rho)), zeros(size(rho)), zeros(size(rho)), zeros(size(rho)))
    end

    Threads.@threads for i in 1:(model.N1)
        for j in 1:(model.N2)
            X = zeros(MVector{4,Float64})
            Grid.ijktoX(i - 1, j - 1, 0, X, model)
            gcov = zeros(MMatrix{4,4,Float64})
            gcon = zeros(MMatrix{4,4,Float64})
            Metrics.gcov_func!(X, model.a, model, gcov)
            Metrics.gcon_func!(gcov, gcon)
            g = Grid.gdet_zone(i - 1, j - 1, 0, model)

            for k in 1:(model.N3)
                Grid.ijktoX(i - 1, j - 1, k, X, model)

                Ufields = (data_array[1].U1, data_array[1].U2, data_array[1].U3)
                UdotU = 0.0
                for l in 1:(Constants.NDIM-1)
                    for m in 1:(Constants.NDIM-1)
                        UdotU += gcov[l+1, m+1] * Ufields[l][i, j, k] * Ufields[m][i, j, k]
                    end
                end

                ufac = sqrt(-1.0 / gcon[1, 1] * (1.0 + abs(UdotU)))
                ucon = MVector{4,Float64}(undef)
                ucon[1] = -ufac * gcon[1, 1]

                for μ in 1:(Constants.NDIM-1)
                    ucon[μ+1] = Ufields[μ][i, j, k] - ufac * gcon[1, μ+1]
                end

                ucov = MVector{4,Float64}(undef)
                Coordinates.flip_index!(ucov, ucon, gcov)
                udotB = 0.0
                Bfields = (data_array[1].B1, data_array[1].B2, data_array[1].B3)
                for l in 1:(Constants.NDIM-1)
                    udotB += ucov[l+1] * Bfields[l][i, j, k]
                end

                bcon = MVector{4,Float64}(undef)
                bcon[1] = udotB
                for μ in 1:(Constants.NDIM-1)
                    bcon[μ+1] = (Bfields[μ][i, j, k] + ucon[μ+1] * udotB) / ucon[1]
                end
                bcov = MVector{4,Float64}(undef)
                Coordinates.flip_index!(bcov, bcon, gcov)

                bsq = 0.0
                for l in 1:Constants.NDIM
                    bsq += bcov[l] * bcon[l]
                end
                data_array[1].b[i, j, k] = sqrt(bsq) * model.B_unit
            end
        end
    end
    init_physical_quantities(data_array, 1, model, trat_large)

    fields = [rho, uu, u1, u2, u3, b1, b2, b3]
    if any(x -> x === nothing, fields)
        @warn "Some primitives were missing in file '$filename'."
    else
        println("All primitives successfully loaded. Dimensions: ", size(rho))
    end

    if advance_path! !== nothing
        advance_path!()
    end
    return data_array[1]
end

function init_physical_quantities(data, n::Int64, model::IharmParams, trat_large::Float64)
    rescale_factor_sqrt = 1.0
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
    beta_arr = data[n].beta
    RHO_arr = data[n].RHO
    UU_arr = data[n].UU
    dθedRhi = data[n].dθedRhi

    @inbounds Threads.@threads for i in 1:model.N1
        for j in 1:model.N2
            for k in 1:model.N3
                rho_ijk = RHO_arr[i, j, k]
                uu_ijk = UU_arr[i, j, k]
                b_ijk = b_arr[i, j, k]

                ne_arr[i, j, k] = rho_ijk * rho_factor

                b_ijk *= rescale_factor_sqrt
                b_arr[i, j, k] = b_ijk

                bsq_normalized = b_ijk * B_unit_inv
                bsq = bsq_normalized * bsq_normalized

                sigma_m = bsq / rho_ijk
                beta_m = uu_ijk * gam_minus_1 / (0.5 * bsq)

                betasq = beta_m * beta_m / beta_crit_sq
                betasq_plus_1_inv = 1.0 / (1.0 + betasq)
                trat = trat_large * betasq * betasq_plus_1_inv + model.trat_small * betasq_plus_1_inv
                θe_unit = θe_factor / (game_minus_1 * trat + gamp_minus_1)
                θe_val = θe_unit * uu_ijk / rho_ijk

                dtratdRhi = betasq * betasq_plus_1_inv
                dθe_unit_dRhi = -θe_factor * game_minus_1 / ((game_minus_1 * trat + gamp_minus_1)^2) * dtratdRhi
                dθe_dRhi = dθe_unit_dRhi * uu_ijk / rho_ijk

                if θe_val > 1.0e-3
                    θe_arr[i, j, k] = θe_val
                    dθedRhi[i, j, k] = dθe_dRhi
                else
                    θe_arr[i, j, k] = 1.0e-3
                    dθedRhi[i, j, k] = 0.0
                end

                sigma_arr[i, j, k] = sigma_m > Constants.SMALL ? sigma_m : Constants.SMALL
                beta_arr[i, j, k] = beta_m > Constants.SMALL ? beta_m : Constants.SMALL
            end
        end
    end
end

function set_tinterp_ns(X, model::IharmParams, data)
    if model.slow_light
        nA = 0
        nB = 0
        if X[1] < data[2].t
            nA = 1
            nB = 2
        else
            nA = 2
            nB = 3
        end
        tinterp = 1.0 - (X[1] - data[nA].t) / (data[nB].t - data[nA].t)
        return nA, nB, tinterp
    else
        return 1, 1, 0.0
    end
end

function get_model_sigma(X, model::IharmParams, data)
    if Grid.X_in_domain(X, model) == 0
        return 0.0
    end
    nA, nB, tfac = set_tinterp_ns(X, model, data)

    return Grid.interp_scalar_time(X, data[nA].sigma, data[nB].sigma, tfac, model.slow_light, model)
end

function get_sigma_smoothfac(sigma, model::IharmParams)
    sigma_above = model.sigma_cut
    if model.sigma_cut_high > 0.0
        sigma_above = model.sigma_cut_high
    end
    if sigma < model.sigma_cut
        return 1.0
    end
    if sigma > sigma_above
        return 0.0
    end
    dsig = sigma_above - model.sigma_cut
    return cos(π / 2.0 / dsig * (sigma - model.sigma_cut))
end

function get_model_ne(X, model::IharmParams, data)
    if Grid.X_in_domain(X, model) == 0
        return 0.0
    end
    sigma_smoothfac = 1.0

    if USE_GEODESIC_SIGMACUT
        sigma = get_model_sigma(X, model, data)
        if sigma > model.sigma_cut
            return 0.0
        end
        sigma_smoothfac = get_sigma_smoothfac(sigma, model)
    end
    nA, nB, tfac = set_tinterp_ns(X, model, data)
    return Grid.interp_scalar_time(X, data[nA].ne, data[nB].ne, tfac, model.slow_light, model) * sigma_smoothfac
end

function get_model_thetae(X, model::IharmParams, data)
    if Grid.X_in_domain(X, model) == 0
        return 0.0
    end
    nA, nB, tfac = set_tinterp_ns(X, model, data)
    return Grid.interp_scalar_time(X, data[nA].θe, data[nB].θe, tfac, model.slow_light, model)
end

function get_model_thetae_deriv(X, model::IharmParams, data)
    if Grid.X_in_domain(X, model) == 0
        return 0.0
    end
    nA, nB, tfac = set_tinterp_ns(X, model, data)
    return Grid.interp_scalar_time(X, data[nA].dθedRhi, data[nB].dθedRhi, tfac, model.slow_light, model)
end

function get_model_b(X, model::IharmParams, data)
    if Grid.X_in_domain(X, model) == 0
        return 0.0
    end
    nA, nB, tfac = set_tinterp_ns(X, model, data)
    return Grid.interp_scalar_time(X, data[nA].b, data[nB].b, tfac, model.slow_light, model)
end

"""
    get_model_fourv!(Ucon, Ucov, Bcon, Bcov, X, Kcon, bhspin, model, data)

Compute the fluid 4-velocity and magnetic field 4-vector at `X`,
interpolated from the GRMHD snapshot(s) in `data`.

`bhspin` is a separate argument (rather than `model.a`) for consistency
with the rest of the differentiable call path.

# Arguments
- `Ucon`, `Ucov`, `Bcon`, `Bcov`: Output vectors, overwritten in place.
- `X`: Position four-vector in internal coordinates.
- `Kcon`: Contravariant photon 4-momentum (unused; kept for interface
  symmetry with the other models' four-velocity functions).
- `bhspin`: Dimensionless black hole spin parameter.
- `model`: Iharm model parameters.
- `data`: GRMHD snapshot(s).
"""
@inline function get_model_fourv!(Ucon, Ucov, Bcon, Bcov, X, Kcon, bhspin, model::IharmParams, data)
    gcov = Metrics.gcov_func(X, bhspin, model)
    gcon = Metrics.gcon_func(gcov)

    if Grid.X_in_domain(X, model) == 0
        Ucov[1] = -1.0 / sqrt(-gcon[1, 1])
        Ucov[2] = 0.0
        Ucov[3] = 0.0
        Ucov[4] = 0.0
        Ucon[1] = 0.0
        Ucon[2] = 0.0
        Ucon[3] = 0.0
        Ucon[4] = 0.0

        for μ in 1:Constants.NDIM
            Ucon[1] += Ucov[μ] * gcon[1, μ]
            Ucon[2] += Ucov[μ] * gcon[2, μ]
            Ucon[3] += Ucov[μ] * gcon[3, μ]
            Ucon[4] += Ucov[μ] * gcon[4, μ]
            Bcon[μ] = 0.0
            Bcov[μ] = 0.0
        end
        return
    end
    Vcon = similar(X)
    nA, nB, tfac = set_tinterp_ns(X, model, data)
    Vcon[2] = Grid.interp_scalar_time(X, data[nA].U1, data[nB].U1, tfac, model.slow_light, model)
    Vcon[3] = Grid.interp_scalar_time(X, data[nA].U2, data[nB].U2, tfac, model.slow_light, model)
    Vcon[4] = Grid.interp_scalar_time(X, data[nA].U3, data[nB].U3, tfac, model.slow_light, model)

    VdotV = 0.0
    for μ in 2:Constants.NDIM
        for ν in 2:Constants.NDIM
            VdotV += gcov[μ, ν] * Vcon[μ] * Vcon[ν]
        end
    end

    Vfac = sqrt(-1.0 / gcon[1, 1] * (1.0 + abs(VdotV)))
    Ucon[1] = -Vfac * gcon[1, 1]
    for μ in 2:Constants.NDIM
        Ucon[μ] = Vcon[μ] - Vfac * gcon[1, μ]
    end

    Ucov_local = Coordinates.flip_index(Ucon, gcov)

    Bcon1 = Grid.interp_scalar_time(X, data[nA].B1, data[nB].B1, tfac, model.slow_light, model)
    Bcon2 = Grid.interp_scalar_time(X, data[nA].B2, data[nB].B2, tfac, model.slow_light, model)
    Bcon3 = Grid.interp_scalar_time(X, data[nA].B3, data[nB].B3, tfac, model.slow_light, model)

    Bcon[1] = (Ucov_local[2] * Bcon1 + Ucov_local[3] * Bcon2 + Ucov_local[4] * Bcon3)
    Bcon[2] = (Bcon1 + Ucon[2] * Bcon[1]) / Ucon[1]
    Bcon[3] = (Bcon2 + Ucon[3] * Bcon[1]) / Ucon[1]
    Bcon[4] = (Bcon3 + Ucon[4] * Bcon[1]) / Ucon[1]

    Bcov_local = Coordinates.flip_index(Bcon, gcov)

    for μ in 1:Constants.NDIM
        Ucov[μ] = Ucov_local[μ]
        Bcov[μ] = Bcov_local[μ]
    end
end

"""
    jar_calc(X, Kcon, bhspin, model, data, derivative_calculation=Val(false))

Compute the emissivity and absorption coefficient of the GRMHD model at
position `X`, optionally with their derivatives with respect to `Rhigh`.

# Arguments
- `X`: Position four-vector in internal coordinates.
- `Kcon`: Contravariant photon 4-momentum.
- `bhspin`: Dimensionless black hole spin parameter.
- `model`: Iharm model parameters.
- `data`: GRMHD snapshot(s).
- `derivative_calculation`: `Val(true)` to additionally return
  `dj/dRhigh`, `dk/dRhigh`.

# Returns
- A tuple `(j, k, dj_dRhigh, dk_dRhigh)`.
"""
function jar_calc(X, Kcon, bhspin, model::IharmParams, data, ::Val{B}=Val(false)) where {B}
    z_base = zero(eltype(X))

    Ne = get_model_ne(X, model, data)
    if Ne == 0.0
        return (z_base, z_base, z_base, z_base)
    end

    elT = promote_type(eltype(X), typeof(bhspin))

    Ucon = MVector{4,elT}(undef)
    Ucov = MVector{4,elT}(undef)
    Bcon = MVector{4,elT}(undef)
    Bcov = MVector{4,elT}(undef)

    get_model_fourv!(Ucon, Ucov, Bcon, Bcov, X, Kcon, bhspin, model, data)
    nu = Radiation.get_fluid_nu(Kcon, Ucov)
    nusq = nu * nu
    θ = Radiation.get_bk_angle(Kcon, Ucov, Bcon, Bcov)
    b = get_model_b(X, model, data)

    θe = get_model_thetae(X, model, data)
    if θ <= 0.0 || θ >= π
        return (z_base, z_base, z_base, z_base)
    end

    j = MaxwellJuettner.maxwell_juettner_I(b, θ, θe, nu, Ne) / nusq
    Bnuinv = Radiation.Bnu_inv(nu, θe)
    z_jk = zero(typeof(j))

    k = (Bnuinv > 0) ? j / Bnuinv : z_jk

    if B
        dθe_dRhigh = get_model_thetae_deriv(X, model, data)
        dj_dθe = ForwardDiff.derivative(θe -> MaxwellJuettner.maxwell_juettner_I(b, θ, θe, nu, Ne) / nusq, θe)

        if Bnuinv > 0
            dk_dθe = (dj_dθe * Bnuinv - j * ForwardDiff.derivative(θe -> Radiation.Bnu_inv(nu, θe), θe)) / (Bnuinv * Bnuinv)
        else
            dk_dθe = z_jk
        end

        dj_dRhigh = dj_dθe * dθe_dRhigh
        dk_dRhigh = dk_dθe * dθe_dRhigh

        return (j, k, dj_dRhigh, dk_dRhigh)
    else
        return (j, k, z_jk, z_jk)
    end
end

"""
    Radiation.get_jk(X, Kcon, freq, bhspin, model::IharmParams, data, derivative_calculation=Val(false))

Compute the emissivity and absorption coefficient of the GRMHD model
(delegates to [`jar_calc`](@ref)).
"""
function Radiation.get_jk(X, Kcon, freq::Float64, bhspin, model::IharmParams, data, derivative_calculation::Val{B}=Val(false)) where {B}
    return jar_calc(X, Kcon, bhspin, model, data, derivative_calculation)
end

"""
    Radiation.radiating_region(X, model::IharmParams, Rh)

Check whether `X` lies within the GRMHD grid's radiating region (within
the grid's radial bounds and away from the polar axis).
"""
function Radiation.radiating_region(X, model::IharmParams, Rh::Float64)
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
    elseif model.metric == Metrics.METRIC_MKS
        th = π * X[3] + ((1.0 - model.hslope) / 2.0) * sin(2.0 * π * X[3])
    else
        error("Unknown METRIC type: $(model.metric)")
    end
    return r, th
end

"""
    Coordinates.bl_coord!(rt, X, model::IharmParams, R0=0.0)

In-place version of [`Coordinates.bl_coord`](@ref) for [`IharmParams`](@ref).
"""
Base.@inline function Coordinates.bl_coord!(rt, X, model::IharmParams, R0::Float64=0.0)
    rt[1] = exp(X[2]) + R0

    if model.metric == Metrics.METRIC_FMKS
        thG = π * X[3] + ((1.0 - model.hslope) / 2.0) * sin(2.0 * π * X[3])
        y = 2 * X[3] - 1.0
        thJ = model.poly_norm * y * (1.0 + ((y / model.poly_xt)^model.poly_alpha) / (model.poly_alpha + 1.0)) + 0.5 * π
        rt[2] = thG + exp(model.mks_smooth * (model.startx[2] - X[2])) * (thJ - thG)
    elseif model.metric == Metrics.METRIC_MKS
        rt[2] = π * X[3] + ((1.0 - model.hslope) / 2.0) * sin(2.0 * π * X[3])
    else
        error("Unknown METRIC type: $(model.metric)")
    end
end

"""
    Coordinates.set_dxdX(X, model::IharmParams)

Compute the Jacobian matrix `dx/dX`, using the FMKS/MKS polar mapping.
"""
function Coordinates.set_dxdX(X, model::IharmParams)
    T = eltype(X)

    dxdX = zeros(MMatrix{4,4,T})

    dxdX[1, 1] = one(T)
    dxdX[4, 4] = one(T)

    dxdX[2, 2] = exp(X[2])

    if model.metric == Metrics.METRIC_MKS
        dxdX[3, 3] = T(π) + (1 - model.hslope) * T(π) * cos(2 * T(π) * X[3])
    elseif model.metric == Metrics.METRIC_FMKS
        dxdX[3, 2] = -exp(model.mks_smooth * (model.startx[2] - X[2])) * model.mks_smooth * (π / 2 - π * X[3] + model.poly_norm * (2 * X[3] - 1) * (1 + ((-1 + 2 * X[3]) / model.poly_xt)^model.poly_alpha / (1 + model.poly_alpha)) - 0.5 * (1 - model.hslope) * sin(2 * π * X[3]))
        dxdX[3, 3] = π + (1 - model.hslope) * π * cos(2 * π * X[3]) + exp(model.mks_smooth * (model.startx[2] - X[2])) * (-π + 2 * model.poly_norm * (1 + ((2 * X[3] - 1) / model.poly_xt)^model.poly_alpha / (model.poly_alpha + 1)) + (2 * model.poly_alpha * model.poly_norm * (2 * X[3] - 1) * ((2 * X[3] - 1) / model.poly_xt)^(model.poly_alpha - 1)) / ((1 + model.poly_alpha) * model.poly_xt) - (1 - model.hslope) * π * cos(2 * π * X[3]))
    else
        error("Unknown METRIC type: $(model.metric)")
    end

    if dxdX[3, 3] <= 0.0
        println("Warning! dxdX[3,3] is non-positive: ", dxdX[3, 3])
        println("X[3] = ", X[3])
        dxdX[3, 3] = T(1.0e-10)
    end

    return SMatrix(dxdX)
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

end
