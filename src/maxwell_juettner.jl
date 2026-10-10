"""
Thermal synchrotron emissivity fitting functions (Maxwell-Juettner
electron distribution).
"""
module MaxwellJuettner

using Bessels
using ForwardDiff
using ..Constants
export get_nu_c, dexter_shape_function, maxwell_juettner_dexter_i, maxwell_juettner_leung_i, maxwell_juettner_i


"""
    besselk_ad(v, x)

`Bessels.besselk(v, x)` that also accepts a `ForwardDiff.Dual` `x` (Bessels.jl does not
support duals), propagating dK_v/dx = -K_{v-1}(x) - (v/x) K_v(x). A private function rather
than a new method of `Bessels.besselk`, so loading Jipole does not change Bessels for other code.
"""
besselk_ad(v, x::Real) = Bessels.besselk(v, x)
function besselk_ad(v, x::ForwardDiff.Dual{T}) where {T}
    val = ForwardDiff.value(x)
    kv = besselk_ad(v, val)
    deriv = -besselk_ad(v - 1, val) - (v / val) * kv
    return ForwardDiff.Dual{T}(kv, deriv * ForwardDiff.partials(x))
end

"""
    get_nu_c(B)

Compute the cyclotron frequency for magnetic field strength `B`.

# Arguments
- `B`: Magnetic field strength (Gauss).

# Returns
- The cyclotron frequency.
"""
function get_nu_c(B)
    return Constants.EE * B / (2 * π * Constants.ME * Constants.CL)
end

"""
    dexter_shape_function(x)

Dexter (2016) fitting function for the thermal synchrotron emissivity
shape. Currently not used, but reminescent from ipole. This should be integrated
as an user choice in the future

# Arguments
- `x`: Dimensionless frequency ratio.

# Returns
- The fitting function value.
"""
function dexter_shape_function(x)
    #TODO: Integrate the dexter fit as an user choice in the future
    return 2.5651 * (1 + 1.92 * x^(-1.0 / 3.0) +
                      0.9977 * x^(-2.0 / 3.0)) * exp(-1.8899 * x^(1.0 / 3.0))
end

"""
    maxwell_juettner_dexter_i(Ne, ν, θe, B, θ)

Dexter (2016) fit for the thermal synchrotron emissivity. Currently not used, but reminescent from ipole. This should be integrated
as an user choice in the future

# Arguments
- `Ne`: Electron number density.
- `ν`: Frequency.
- `θe`: Dimensionless electron temperature.
- `B`: Magnetic field strength.
- `θ`: Angle between the photon wavevector and the magnetic field.

# Returns
- The emissivity (erg/s/cm^3).
"""
function maxwell_juettner_dexter_i(Ne, ν, θe, B, θ)
    nus = 3.0 * Constants.EE * B * sin(θ) / 4.0 / π / Constants.ME / Constants.CL * θe * θe + 1.0
    x = ν / nus

    j = Ne * Constants.EE * Constants.EE * ν / 2.0 / sqrt(3) / Constants.CL / θe / θe * dexter_shape_function(x)

    if isnan(j) || isinf(j)
        println("j nan in Dexter fit: j $j x $x nu $ν nus $nus Thetae $θe")
    end
    return j
end

"""
    maxwell_juettner_leung_i(Ne, ν, θe, B, θ)

Leung et al. (2011) fit for the thermal synchrotron emissivity.

# Arguments
- `Ne`: Electron number density.
- `ν`: Frequency.
- `θe`: Dimensionless electron temperature.
- `B`: Magnetic field strength.
- `θ`: Angle between the photon wavevector and the magnetic field.

# Returns
- The emissivity (erg/s/cm^3).
"""
function maxwell_juettner_leung_i(Ne, ν, θe, B, θ)
    T = promote_type(typeof(Ne), typeof(ν), typeof(θe), typeof(B), typeof(θ))
    K2 = max(besselk_ad(2, 1.0 / θe), T(Constants.SMALL))
    nuc = Constants.EE * B / (2.0 * π * Constants.ME * Constants.CL)
    nus = (2.0 / 9.0) * nuc * θe * θe * sin(θ)
    if ν > 1.e12 * nus
        return zero(T)
    end
    x = ν / nus
    f = (x^(1.0 / 2.0) + 2.0^(11.0 / 12.0) * x^(1.0 / 6.0))^2
    j = (sqrt(2.0) * π * Constants.EE^2 * Ne * nus / (3.0 * Constants.CL * K2)) * f * exp(-x^(1.0 / 3.0))
    return j
end

function maxwell_juettner_i(B, θ, θe, ν, ne)
    return maxwell_juettner_leung_i(ne, ν, θe, B, θ)
end

end
