"""
Analytic torus emission model (Gold et al. 2020, Section 3.2 test problem).
"""
module Analytic

using StaticArrays
using ..Constants
using ..AbstractModels
using ..Coordinates
using ..Metrics
using ..Radiation

export AnalyticParams, get_analytic_jk

"""
Parameters for the [`Analytic`](@ref) torus emission model.

# Fields
- `a`: Dimensionless black hole spin parameter.
- `Rout`: Outer radius of the computational domain.
- `cstartx`, `cstopx`: Native coordinate grid bounds.
- `metric`: Metric family (always `Metrics.METRIC_MKS` for this model).
- `hslope`: Polar coordinate compression (always `1.0`, i.e. uncompressed,
  for this model).
- `A`: Absorption coefficient normalization.
- `α`: Emissivity's exponential dependence on frequency (spectral index).
- `height`: Disk height parameter defining the vertical structure of the
  torus.
- `l0`: Normalization of the specific angular momentum profile.
- `L_unit`, `T_unit`: Length/time units (cm, s), set by `MBH`.
- `RHO_unit`, `U_unit`, `B_unit`: Density/internal-energy/magnetic-field
  units (g/cm^3, erg/cm^3, Gauss).
"""
struct AnalyticParams{T} <: AbstractModel
    a::T
    Rout::Float64
    cstartx::MVector{4,Float64}
    cstopx::MVector{4,Float64}
    metric::Int
    hslope::Float64
    A::Float64
    α::Float64
    height::Float64
    l0::Float64
    L_unit::Float64
    T_unit::Float64
    RHO_unit::Float64
    U_unit::Float64
    B_unit::Float64
    rmax_geo::T
end

"""
    AnalyticParams(bhspin, Rout, cstartx, cstopx, rmax_geo, MBH; A=1.e6, α=-0.0, height=100.0/3.0, l0=1.0, RHO_unit=3.e-18)

Construct [`AnalyticParams`](@ref), deriving the length/time/density units
from the black hole mass `MBH`.

# Arguments
- `bhspin`: Dimensionless black hole spin parameter.
- `Rout`: Outer radius of the computational domain.
- `cstartx`, `cstopx`: Native coordinate grid bounds.
- `MBH`: Black hole mass, in solar masses.
- `A`: Absorption coefficient normalization.
- `α`: Emissivity's exponential dependence on frequency (spectral index).
- `height`: Disk height parameter.
- `l0`: Normalization of the specific angular momentum profile.
- `RHO_unit`: Density unit, in g/cm^3.
"""
function AnalyticParams(bhspin, Rout, cstartx, cstopx, rmax_geo, MBH; A=1.e6, α=-0.0, height=100.0 / 3.0, l0=1.0, RHO_unit=3.e-18)
    L_unit = Constants.GNEWT * MBH * Constants.MSUN / Constants.CL^2
    T_unit = L_unit / Constants.CL
    U_unit = RHO_unit * Constants.CL^2
    B_unit = Constants.CL * sqrt(4 * π * RHO_unit)
    return AnalyticParams(bhspin, Rout, MVector{4,Float64}(cstartx), MVector{4,Float64}(cstopx),
        Metrics.METRIC_MKS, 1.0, A, α, height, l0, L_unit, T_unit, RHO_unit, U_unit, B_unit, rmax_geo)
end

"""
    Radiation.radiating_region(X, model::AnalyticParams, Rh)

Check whether `X` lies within the analytic torus's radiating region
(a fixed radial shell around the black hole).

# Arguments
- `X`: Position four-vector in internal coordinates.
- `model`: Analytic model parameters.
- `Rh`: Event horizon radius.

# Returns
- `true` if `X` is within the radiating region.
"""
function Radiation.radiating_region(X, model::AnalyticParams, Rh::Float64)
    r, _ = Coordinates.bl_coord(X, model)
    return r > (Rh + 0.0001) && r > 1.0 && r < 1000.0
end

"""
    get_model_ne(X, model)

Compute the electron number density of the analytic torus at `X`.

Follows the model described in Gold et al. (2020),
https://iopscience.iop.org/article/10.3847/1538-4357/ab96c6.

# Arguments
- `X`: Position four-vector in internal coordinates.
- `model`: Analytic model parameters.

# Returns
- The electron number density.
"""
function get_model_ne(X, model::AnalyticParams)
    r, th = Coordinates.bl_coord(X, model)

    n_exp = 0.5 * ((r / 10)^2 + (model.height * cos(th))^2)
    return (n_exp < 200) ? model.RHO_unit * exp(-n_exp) : 0.0
end

"""
    get_model_4vel(X, bhspin, model)

Compute the covariant fluid 4-velocity of the analytic torus at `X`.

Follows the model described in Gold et al. (2020),
https://iopscience.iop.org/article/10.3847/1538-4357/ab96c6.

`bhspin` is a separate argument (rather than `model.a`) so this function
stays differentiable when called on the autodiff path.

# Arguments
- `X`: Position four-vector in internal coordinates.
- `bhspin`: Dimensionless black hole spin parameter.
- `model`: Analytic model parameters.

# Returns
- The covariant fluid 4-velocity `Ucov`.
"""
function get_model_4vel(X, bhspin, model::AnalyticParams)
    r, th = Coordinates.bl_coord(X, model)
    R = r * sin(th)
    l = model.l0 / (1 + R) * R^(1.5)
    T = promote_type(typeof(r), typeof(th), typeof(bhspin))
    bl_gcov = zeros(MMatrix{4,4,T})
    bl_gcon = zeros(MMatrix{4,4,T})
    bl_Ucov = zeros(MVector{4,T})

    Metrics.gcov_bl!(r, th, bhspin, bl_gcov)
    Metrics.gcon_func!(bl_gcov, bl_gcon)

    gcov = Metrics.gcov_func(X, bhspin, model)

    ubar = sqrt(-1.0 / (bl_gcon[1, 1] - 2.0 * bl_gcon[1, 4] * l + bl_gcon[4, 4] * l * l))
    bl_Ucov[1] = -ubar
    bl_Ucov[2] = zero(T)
    bl_Ucov[3] = zero(T)
    bl_Ucov[4] = l * ubar
    bl_Ucon = Coordinates.flip_index(bl_Ucov, bl_gcon)

    ks_Ucon = Coordinates.bl_to_ks(X, bl_Ucon, bhspin, model)
    Ucon = Coordinates.vec_from_ks(X, ks_Ucon, model)
    Ucov = Coordinates.flip_index(Ucon, gcov)

    return Ucov
end

"""
    get_analytic_jk(X, Kcon, freqcgs, bhspin, model)

Compute the emissivity and absorption coefficient of the analytic torus
at position `X` and frequency `freqcgs`.

`bhspin` is a separate argument (rather than `model.a`) so this function
stays differentiable when called on the autodiff path.

# Arguments
- `X`: Position four-vector in internal coordinates.
- `Kcon`: Contravariant photon 4-momentum, in internal coordinates.
- `freqcgs`: Pivotal frequency, in cgs units.
- `bhspin`: Dimensionless black hole spin parameter.
- `model`: Analytic model parameters.

# Returns
- A tuple `(j, k, 0, 0)`.
"""
function get_analytic_jk(X, Kcon, freqcgs::Float64, bhspin, model::AnalyticParams)
    Ne = get_model_ne(X, model)
    if Ne <= 0.0
        z = zero(eltype(X))
        return z, z, z, z
    end

    Ucov = get_model_4vel(X, bhspin, model)
    ν = Radiation.get_fluid_nu(Kcon, Ucov)

    if ν <= 0.0 || any(isnan.(ν)) || any(isinf.(ν))
        println("At X = $X\n Kcon = $Kcon")
        println("Ucov = $Ucov")
        println("Kcon $Kcon")
        error("Frequency must be positive, got ν = $ν")
    end

    jnu_inv = max(Ne * (ν / freqcgs)^(-model.α) / ν^2, 0.0)
    knu_inv = max((model.A * Ne * (ν / freqcgs)^(-(model.α + 2.5)) + 1.e-54) * ν, 0.0)

    if isnan(jnu_inv) || isinf(jnu_inv)
        @error "Invalid jnu_inv computed" jnu_inv
        println("Ne = $Ne, ν = $ν")
        error("Invalid jnu_inv computed: $jnu_inv")
    end
    if isnan(knu_inv) || isinf(knu_inv)
        @error "Invalid knu_inv computed" knu_inv
        println("Ne = $Ne, ν = $ν")
        error("Invalid knu_inv computed: $knu_inv")
    end

    z = zero(typeof(jnu_inv))
    return jnu_inv, knu_inv, z, z
end

"""
    Radiation.get_jk(X, Kcon, freq, bhspin, model::AnalyticParams, data=nothing, derivative_calculation=Val(false))

Compute the emissivity and absorption coefficient of the analytic torus
(delegates to [`get_analytic_jk`](@ref); `data` and
`derivative_calculation` are unused by this model).
"""
function Radiation.get_jk(X, Kcon, freq::Float64, bhspin, model::AnalyticParams, data=nothing, derivative_calculation::Val{B}=Val(false)) where {B}
    return get_analytic_jk(X, Kcon, freq, bhspin, model)
end

end
