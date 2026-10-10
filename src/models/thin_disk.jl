"""
Thin disk (Novikov-Thorne / Krolik & Hawley) boundary-emission model.
"""
module ThinDisk

using StaticArrays
using ..Constants
using ..GeoTypes
using ..AbstractModels
using ..Coordinates
using ..Metrics
using ..Radiation

export ThinDiskParams

"""
Parameters for the [`ThinDisk`](@ref) boundary-emission model.

# Fields
- `a`: Dimensionless black hole spin parameter.
- `Rout`: Outer radius of the computational domain.
- `cstartx`, `cstopx`: Native coordinate grid bounds.
- `metric`: Metric family (always `Metrics.METRIC_MKS` for this model).
- `hslope`: Polar coordinate compression (always `1.0`, i.e. uncompressed,
  for this model).
- `f`: Spectral hardening factor.
- `r_isco`: Innermost stable circular orbit radius, for spin `a`.
- `T0`: Disk temperature normalization.
- `L_unit`, `T_unit`: Length/time units (cm, s), set by `MBH`.
"""
struct ThinDiskParams <: AbstractModel
    a::Float64
    Rout::Float64
    cstartx::MVector{4,Float64}
    cstopx::MVector{4,Float64}
    metric::Int
    hslope::Float64
    f::Float64
    r_isco::Float64
    T0::Float64
    L_unit::Float64
    T_unit::Float64
    rmax_geo::Float64
end

"""
    ThinDiskParams(bhspin, Rout, cstartx, cstopx, MBH, Mdot; f=1.8)

Construct [`ThinDiskParams`](@ref), deriving the length/time units and the
disk temperature normalization from the black hole mass `MBH` and the
mass accretion rate `Mdot`.

# Arguments
- `bhspin`: Dimensionless black hole spin parameter.
- `Rout`: Outer radius of the computational domain.
- `cstartx`, `cstopx`: Native coordinate grid bounds.
- `MBH`: Black hole mass, in solar masses.
- `Mdot`: Mass accretion rate, in g/s.
- `f`: Spectral hardening factor.
"""
function ThinDiskParams(bhspin, Rout, cstartx, cstopx, MBH, Mdot, Rstop; f=1.8)
    L_unit = Constants.GNEWT * MBH * Constants.MSUN / Constants.CL^2
    T_unit = L_unit / Constants.CL

    z1 = 1.0 + (1.0 - bhspin^2)^(1.0 / 3.0) * ((1.0 + bhspin)^(1.0 / 3.0) + (1.0 - bhspin)^(1.0 / 3.0))
    z2 = sqrt(3.0 * bhspin^2 + z1^2)
    r_isco = 3.0 + z2 - copysign(sqrt((3.0 - z1) * (3.0 + z1 + 2.0 * z2)), bhspin)
    T0 = (3.0 / (8.0) / π * Constants.GNEWT * MBH * Constants.MSUN * Mdot / L_unit^3 / Constants.SIG)^(1.0 / 4.0)

    return ThinDiskParams(bhspin, Rout, MVector{4,Float64}(cstartx), MVector{4,Float64}(cstopx),
        Metrics.METRIC_MKS, 1.0, f, r_isco, T0, L_unit, T_unit, Rstop)
end

"""
    thindisk_region(Xi, Xf, model)

Check whether the geodesic segment from `Xi` to `Xf` crosses the thin
disk's midplane within its emitting radial range.

# Arguments
- `Xi`: Starting position four-vector, in internal coordinates.
- `Xf`: Ending position four-vector, in internal coordinates.
- `model`: Thin disk model parameters.

# Returns
- `true` if the segment crosses the emitting region of the disk.
"""
function thindisk_region(Xi::MVector{4,Float64}, Xf::MVector{4,Float64}, model::ThinDiskParams)::Bool
    _, th_i = Coordinates.bl_coord(Xi, model)
    r_f, th_f = Coordinates.bl_coord(Xf, model)

    midplane::Bool = (sign(th_i - π / 2) != sign(th_f - π / 2))
    em_region::Bool = (r_f > model.r_isco && r_f < model.Rout)
    return midplane && em_region
end

"""
    Radiation.radiating_region(X, model::ThinDiskParams, Rh)

Always `false`: the thin disk model has no volumetric emission, only
boundary-crossing emission (see [`thindisk_region`](@ref) /
[`get_td_boundary_condition`](@ref)).
"""
function Radiation.radiating_region(X, model::ThinDiskParams, Rh::Float64)
    return false
end

"""
    get_td_boundary_condition(X, Kcon, bhspin, Rh, model)

Compute the disk-surface intensity where the geodesic crosses the thin
disk.

# Arguments
- `X`: Position four-vector in internal coordinates, at the crossing.
- `Kcon`: Contravariant photon 4-momentum, at the crossing.
- `bhspin`: Dimensionless black hole spin parameter.
- `Rh`: Event horizon radius.
- `model`: Thin disk model parameters.

# Returns
- The disk-surface intensity, or `0.0` if inside the event horizon.
"""
function get_td_boundary_condition(X, Kcon, bhspin, Rh::Float64, model::ThinDiskParams)
    r, _ = Coordinates.bl_coord(X, model)

    if r > Rh
        Temp, omega = thindisk_vals(r, bhspin, model)

        Ucon, Ucov, Bcon, Bcov = get_model_fourv(X, bhspin, model)

        mu = abs(cos(Radiation.get_bk_angle(Kcon, Ucov, Bcon, Bcov)))

        nu = Radiation.get_fluid_nu(Kcon, Ucov)

        if nu <= 0.0 || any(isnan(nu)) || any(isinf(nu))
            println("At X = $X\n Kcon = $Kcon")
            println("Ucov = $Ucov")
            println("Kcon $Kcon")
            error("Frequency must be positive, got nu = $nu")
        end

        I = fbbpolemis!(nu, Temp, mu, model)
    else
        I = 0.0
    end
    return I
end

"""
    thindisk_vals(r, bhspin, model)

Compute the local disk temperature and orbital angular frequency at
radius `r`.

# Arguments
- `r`: Radial coordinate, in Boyer-Lindquist/Kerr-Schild coordinates.
- `bhspin`: Dimensionless black hole spin parameter.
- `model`: Thin disk model parameters.

# Returns
- A tuple `(Temp, omega)`.
"""
function thindisk_vals(r::Float64, bhspin::Float64, model::ThinDiskParams)
    b = 1.0 - 3.0 / r + 2.0 * bhspin / r^(3 / 2)
    kc = krolikc(r, bhspin, model)
    d = r * r - 2.0 * r + bhspin * bhspin
    lc = (model.r_isco * model.r_isco - 2.0 * bhspin * sqrt(model.r_isco) + bhspin * bhspin) / (sqrt(model.r_isco) - 2.0 * sqrt(model.r_isco) + bhspin)
    hc = (2.0 * r - bhspin * lc) / d
    ar = (r * r + bhspin * bhspin)^2 - bhspin * bhspin * d * sin(π / 2.0)^2
    om = 2.0 * bhspin * r / ar

    if r > model.r_isco
        omega = max(1.0 / (r^(3 / 2) + bhspin), om)
    else
        omega = max((lc + bhspin * hc) / (r * r + 2.0 * r * (1.0 + hc)), om)
    end

    if r > model.r_isco && r < model.Rout
        Temp = model.T0 * (kc / b / r^3)^(1.0 / 4.0)
    else
        Temp = model.T0 / 1.e5
    end
    return Temp, omega
end

"""
    krolikc(r, bhspin, model)

Compute the Krolik & Hawley (2002) relativistic correction factor for the
disk temperature profile.

# Arguments
- `r`: Radial coordinate, in Boyer-Lindquist/Kerr-Schild coordinates.
- `bhspin`: Dimensionless black hole spin parameter.
- `model`: Thin disk model parameters.

# Returns
- The Krolik & Hawley correction factor.
"""
function krolikc(r::Float64, bhspin::Float64, model::ThinDiskParams)
    y = sqrt(r)
    yms = sqrt(model.r_isco)
    y1 = 2.0 * cos(1.0 / 3.0 * (acos(bhspin) - π))
    y2 = 2.0 * cos(1.0 / 3.0 * (acos(bhspin) + π))
    y3 = -2.0 * cos(1.0 / 3.0 * acos(bhspin))
    arg1 = 3.0 * bhspin / (2.0 * y)
    arg2 = 3.0 * (y1 - bhspin)^2 / (y * y1 * (y1 - y2) * (y1 - y3))
    arg3 = 3.0 * (y2 - bhspin)^2 / (y * y2 * (y2 - y1) * (y2 - y3))
    arg4 = 3.0 * (y3 - bhspin)^2 / (y * y3 * (y3 - y1) * (y3 - y2))

    return 1.0 - yms / y - arg1 * log(y / yms) - arg2 * log((y - y1) / (yms - y1)) - arg3 * log((y - y2) / (yms - y2)) - arg4 * log((y - y3) / (yms - y3))
end

"""
    fbbpolemis!(nu, Temp, cosne, model)

Compute the disk-surface emissivity for the thin disk model.

# Arguments
- `nu`: Frequency, in the fluid frame.
- `Temp`: Local disk temperature.
- `cosne`: Cosine of the angle between the photon direction and the
  magnetic field.
- `model`: Thin disk model parameters.

# Returns
- The emissivity.
"""
function fbbpolemis!(nu::Float64, Temp::Float64, cosne::Float64, model::ThinDiskParams)
    I = model.f^(-4.0) * bnu(nu, Temp * model.f)

    if nu <= 0.0 || any(isnan(nu)) || any(isinf(nu))
        error("Frequency must be positive, got nu = $nu")
    end
    interpI, interpDel = interp_chandra(cosne)

    I *= interpI / (nu * nu * nu)

    return I
end

"""
    bnu(nu, Temp)

Compute the Planck function for a given frequency and temperature.

# Arguments
- `nu`: Frequency.
- `Temp`: Temperature.

# Returns
- The Planck function value.
"""
function bnu(nu::Float64, Temp::Float64)
    return 2 * Constants.HPL * nu^3 / (Constants.CL^2) / (exp(Constants.HPL * nu / (Constants.KBOL * Temp)) - 1)
end

const CH_TABLE = let
    fname = joinpath(@__DIR__, "..", "..", "Tables", "ch24_vals.txt")
    if !isfile(fname)
        @error "Error reading file $fname!"
        error("File not found: $fname")
    end
    npts = 21
    ch_mu = zeros(npts)
    ch_I = zeros(npts)
    ch_delta = zeros(npts)
    open(fname, "r") do vals
        for i in 1:npts
            line = readline(vals)
            vals_split = split(line)
            ch_mu[i] = parse(Float64, vals_split[1])
            ch_I[i] = parse(Float64, vals_split[2])
            ch_delta[i] = parse(Float64, vals_split[3])
        end
    end
    @info "Chandra table loaded successfully from $fname"
    (mu=ch_mu, I=ch_I, delta=ch_delta)
end

"""
    get_weight!(xx, x, jlo)

Locate the interpolation weight and bracketing index of `x` within the
sorted table `xx`, starting the search at `jlo`.

# Arguments
- `xx`: Sorted vector of tabulated x-values.
- `x`: Value to interpolate at.
- `jlo`: Initial guess for the lower bracketing index.

# Returns
- A tuple `(weight, jlo)`.
"""
function get_weight!(xx::Vector{Float64}, x::Float64, jlo::Int64)
    while xx[jlo] < x
        jlo += 1
    end
    jlo -= 1
    return (x - xx[jlo]) / (xx[jlo+1] - xx[jlo]), jlo
end

"""
    interp_chandra(mu)

Interpolate the Chandrasekhar (1960) limb-darkening emissivity and
polarization tables at `mu`.

# Arguments
- `mu`: Cosine of the angle between the photon direction and the disk
  normal.

# Returns
- A tuple `(I, delta)`.
"""
function interp_chandra(mu::Float64)
    indx::Int64 = 1
    weight, indx = get_weight!(CH_TABLE.mu, mu, indx)
    i = (1.0 - weight) * CH_TABLE.I[indx] + weight * CH_TABLE.I[indx+1]
    del = (1.0 - weight) * CH_TABLE.delta[indx] + weight * CH_TABLE.delta[indx+1]
    return i, del
end

"""
    get_model_fourv(X, bhspin, model)

Compute the fluid 4-velocity and magnetic field 4-vector at the thin
disk's surface at `X`.

`bhspin` is a separate argument (rather than `model.a`) for consistency
with the rest of the differentiable call path.

# Arguments
- `X`: Position four-vector in internal coordinates.
- `bhspin`: Dimensionless black hole spin parameter.
- `model`: Thin disk model parameters.

# Returns
- A tuple `(Ucon, Ucov, Bcon, Bcov)`.
"""
@inline function get_model_fourv(X::MVector{4,Float64}, bhspin::Float64, model::ThinDiskParams)
    gcov = Metrics.gcov_func(X, bhspin, model)
    r, _ = Coordinates.bl_coord(X, model)
    _, omega = thindisk_vals(r, bhspin, model)
    Ucon = MVector{4,Float64}(undef)
    Ucon[1] = sqrt(-1.0 / (gcov[1, 1] + 2.0 * gcov[1, 4] * omega + gcov[4, 4] * omega * omega))
    Ucon[2] = 0.0
    Ucon[3] = 0.0
    Ucon[4] = omega * Ucon[1]
    Ucov = Coordinates.flip_index(Ucon, gcov)
    Bcon = calc_polvec(X, bhspin, model)
    Bcov = Coordinates.flip_index(Bcon, gcov)
    return Ucon, Ucov, Bcon, Bcov
end

"""
    calc_polvec(X, bhspin, model)

Compute the disk-surface polarization vector in Kerr-Schild coordinates.

# Arguments
- `X`: Position four-vector in internal coordinates.
- `bhspin`: Dimensionless black hole spin parameter.
- `model`: Thin disk model parameters.

# Returns
- The (normalized) polarization 4-vector.
"""
function calc_polvec(X::MVector{4,Float64}, bhspin::Float64, model::ThinDiskParams)
    fourf_bl = zeros(MVector{4,Float64})
    fourf_bl[3] = 1.0
    fourf_ks = Coordinates.bl_to_ks(X, fourf_bl, bhspin, model)
    fourf = Coordinates.vec_to_ks(X, fourf_ks, model)

    gcov = Metrics.gcov_func(X, bhspin, model)
    fourf_cov = Coordinates.flip_index(fourf, gcov)

    normf = sqrt(fourf[1] * fourf_cov[1] +
                 fourf[2] * fourf_cov[2] +
                 fourf[3] * fourf_cov[3] +
                 fourf[4] * fourf_cov[4])

    for i in 1:4
        fourf[i] /= normf
    end

    return fourf
end

"""
    Radiation.integrate_emission!(traj, nsteps, Image, I, J, freq, bhspin, model::ThinDiskParams, data=nothing)

Integrate the emission along the geodesic trajectory, using the thin
disk's boundary-crossing emission model rather than a volumetric
integral: `Image[I, J]` is set to the surface intensity of the last
disk-midplane crossing found along the trajectory (`data` is unused by
this model).
"""
function Radiation.integrate_emission!(traj::Vector{GeoTypes.OfTrajGeneric{T}}, nsteps::Int, Image, I, J, freq, bhspin, model::ThinDiskParams, data=nothing) where {T}
    Xi = MVector{4,T}(undef)
    Kconi = MVector{4,T}(undef)
    Xf = MVector{4,T}(undef)
    Rh = 1 + sqrt(1.0 - bhspin * bhspin)

    Intensity = zero(T)
    for nstep = nsteps:-1:2
        for k in 1:Constants.NDIM
            Xi[k] = traj[nstep].X[k]
            Xf[k] = traj[nstep-1].X[k]
            Kconi[k] = traj[nstep].Kcon[k]
        end
        if thindisk_region(Xi, Xf, model)
            Intensity = get_td_boundary_condition(Xi, Kconi, bhspin, Rh, model)
        end
    end
    Image[I, J] = Intensity
end

end
