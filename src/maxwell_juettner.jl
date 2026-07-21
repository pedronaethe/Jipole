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
    Bessels.besselk(v::Real, x::ForwardDiff.Dual{T,V,N}) where {T,V,N}

Compute the modified Bessel function of the second kind for a ForwardDiff.Dual number. 
This is a patch because the Bessels.jl package does not natively support ForwardDiff.Dual types.

# Arguments
- `v`: Order of the Bessel function.
- `x`: Input value, which can be a ForwardDiff.Dual number.

# Returns
- The value of the modified Bessel function of the second kind evaluated at `x` and its derivative with respect to `x`.
"""
function Bessels.besselk(v::Real, x::ForwardDiff.Dual{T,V,N}) where {T,V,N}
    val = ForwardDiff.value(x)
    partials = ForwardDiff.partials(x)
    kv = Bessels.besselk(v, val)
    kv_minus = Bessels.besselk(v - 1, val)
    deriv = -kv_minus - (v / val) * kv
    return ForwardDiff.Dual{T}(kv, deriv * partials)
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
shape.

# Arguments
- `x`: Dimensionless frequency ratio.

# Returns
- The fitting function value.
"""
function dexter_shape_function(x)
    return 2.5651 * (1 + 1.92 * x^(-1.0 / 3.0) +
                      0.9977 * x^(-2.0 / 3.0)) * exp(-1.8899 * x^(1.0 / 3.0))
end

"""
    maxwell_juettner_dexter_i(Ne, ν, θe, B, θ)

Dexter (2016) fit for the thermal synchrotron emissivity.

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
    K2 = max(besselk(2, 1.0 / θe), T(Constants.SMALL))
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

"""
    maxwell_juettner_i(B, θ, θe, ν, ne)

Thermal synchrotron emissivity used by the radiative transfer.

Currently always uses the Leung et al. (2011) fit ([`maxwell_juettner_leung_i`](@ref)),
since it is unpolarized; the Dexter (2016) fit below is kept for when
polarized transport is added (`ipole`'s C implementation selects the
Dexter fit via `params.dexter_fit = 1`).

# Arguments
- `B`: Magnetic field strength.
- `θ`: Angle between the photon wavevector and the magnetic field.
- `θe`: Dimensionless electron temperature.
- `ν`: Frequency.
- `ne`: Electron number density.

# Returns
- The emissivity (erg/s/cm^3).
"""
function maxwell_juettner_i(B, θ, θe, ν, ne)
    #TODO: For now, we are gonna return leung as it's non polarized, but original IPOLE code implementation has dexter params.dexter_fit = 1 for polarized transport
    return maxwell_juettner_leung_i(ne, ν, θe, B, θ)

    νc = get_nu_c(B)

    νs = (2.0 / 9.0) * νc * θe * θ^2

    X = ν / νs

    prefactor = ne * Constants.EE^2 * νc / (Constants.CL)

    term1 = sqrt(2.0) * π / 27.0 * sin(θ)
    term2 = ((X^(1.0 / 2.0)) + 2.0^(11.0 / 12.0) * X^(1.0 / 6.0))^2
    term3 = exp(-X^(1.0 / 3.0))

    ans = prefactor * term1 * term2 * term3

    if isnan(ans) || isinf(ans)
        println("Invalid maxwell_juettner_i calculation:")
        println("B = $B, θ = $θ, θe = $θe, ν = $ν, ne = $ne")
        println("Computed values: νc = $νc, νs = $νs, X = $X, prefactor = $prefactor, term1 = $term1, term2 = $term2, term3 = $term3")
        error("Resulting intensity is NaN or Inf")
    end
    return ans
end

end
