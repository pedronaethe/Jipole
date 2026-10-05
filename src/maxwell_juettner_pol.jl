"""
Polarized transfer coefficients for a relativistic thermal (Maxwell-Juettner)
electron distribution: the Stokes Q and V synchrotron emissivities and the
Faraday conversion/rotation coefficients `rho_Q`, `rho_V`.

This is the polarized counterpart of `maxwell_juettner.jl` and a translation of
the thermal routines of ipole's `src/symphony/maxwell_juettner_fits.c`:

| here                              | ipole                                         |
|-----------------------------------|-----------------------------------------------|
| `dexter_shape_function_q`         | `I_Q`                                         |
| `dexter_shape_function_v`         | `I_V`                                         |
| `maxwell_juettner_dexter_iqv`     | `maxwell_juettner_dexter_I/_Q/_V`             |
| `bessel_k_ratios`                 | `gsl_sf_bessel_Kn` + `(k2 > 0) ? k/k2 : 1`    |
| `maxwell_juettner_rho_q`          | `maxwell_juettner_rho_Q`                      |
| `maxwell_juettner_rho_v`          | `maxwell_juettner_rho_V` (`dexter_fit == 0`)  |
| `maxwell_juettner_rho_v_dexter`   | `maxwell_juettner_rho_V` (`dexter_fit != 0`)  |

Every function here returns the coefficient in the *fluid frame, in CGS* and with
the *sign convention of the fit* (ipole's "Symphony convention"), exactly like the
C routine it mirrors. The conversion to invariant quantities, the sign flips to the
tetrad convention and the consistency caps are done one level up, in
`Polarization.thermal_jar` (ipole's `jar_calc_dist`).

All routines are pure functions of plain numbers: no allocation, no I/O and no
exceptions on the code paths used by the transfer loop, so they can be called from
a GPU kernel, and they are generic in the number type so `ForwardDiff.Dual`s pass
through.
"""
module MaxwellJuettnerPol

using Bessels
using ForwardDiff
using ..Constants
using ..MaxwellJuettner

export dexter_shape_function_q, dexter_shape_function_v, maxwell_juettner_dexter_iqv,
    bessel_k_ratios, maxwell_juettner_rho_q, maxwell_juettner_rho_v, maxwell_juettner_rho_v_dexter

"""
    dexter_shape_function_q(x)

Dexter (2016) fitting function `I_Q(x)` for the Stokes Q thermal synchrotron
emissivity (ipole's `I_Q`). The Stokes I counterpart `I_I(x)` already exists as
`MaxwellJuettner.dexter_shape_function`.

# Arguments
- `x`: Frequency ratio `nu / nu_s`.

# Returns
- The value of `I_Q(x)`.
"""
@inline function dexter_shape_function_q(x)
    return 2.5651 * (1 + 0.93193 * x^(-1.0 / 3.0) +
                      0.499873 * x^(-2.0 / 3.0)) * exp(-1.8899 * x^(1.0 / 3.0))
end

"""
    dexter_shape_function_v(x)

Dexter (2016) fitting function `I_V(x)` for the Stokes V thermal synchrotron
emissivity (ipole's `I_V`).

# Arguments
- `x`: Frequency ratio `nu / nu_s`.

# Returns
- The value of `I_V(x)`.
"""
@inline function dexter_shape_function_v(x)
    return (1.81384 / x + 3.42319 * x^(-2.0 / 3.0) +
            0.0292545 * x^(-0.5) + 2.03773 * x^(-1.0 / 3.0)) * exp(-1.8899 * x^(1.0 / 3.0))
end

"""
    maxwell_juettner_dexter_iqv(Ne, ν, θe, B, θ)

Thermal synchrotron emissivities in Stokes I, Q and V from the Dexter (2016)
fits. This evaluates ipole's `maxwell_juettner_dexter_I`, `_Q` and `_V` in one
call, since the three share `nu_s`, `x` and most of the prefactor.

The Stokes I value is the same quantity `MaxwellJuettner.maxwell_juettner_dexter_i`
returns; that routine is not called here because its diagnostic `println` cannot be
compiled into a GPU kernel.

Sign convention: as in ipole's fit routines, i.e. `jQ <= 0` ("Symphony
convention"). `Polarization.thermal_jar` flips it to the tetrad convention.

# Arguments
- `Ne`: Electron number density [cm^-3].
- `ν`: Frequency in the fluid frame [Hz].
- `θe`: Dimensionless electron temperature.
- `B`: Magnetic field strength [G].
- `θ`: Angle between the wavevector and the magnetic field [rad]. Must lie
  strictly inside `(0, π)`; the field-aligned case is handled by the caller.

# Returns
- A tuple `(jI, jQ, jV)` in CGS (erg s^-1 cm^-3 Hz^-1 sr^-1).
"""
@inline function maxwell_juettner_dexter_iqv(Ne, ν, θe, B, θ)
    # The "+ 1.0" (one Hz) is ipole's guard against nus == 0.
    nus = 3.0 * Constants.EE * B * sin(θ) / 4.0 / π / Constants.ME / Constants.CL * θe * θe + 1.0
    x = ν / nus

    # Common prefactor of jI and jQ; operation order follows ipole.
    pref = Ne * Constants.EE * Constants.EE * ν / 2.0 / sqrt(3) / Constants.CL / θe / θe

    jI = pref * MaxwellJuettner.dexter_shape_function(x)
    jQ = -pref * dexter_shape_function_q(x)
    jV = 2.0 * Ne * Constants.EE * Constants.EE * ν / tan(θ) / 3.0 / sqrt(3) / Constants.CL / θe / θe / θe * dexter_shape_function_v(x)

    return jI, jQ, jV
end

"""
    _primal(x)

Strip every level of `ForwardDiff.Dual` from `x`, returning the plain number.
"""
@inline _primal(x::Real) = x
@inline _primal(x::ForwardDiff.Dual) = _primal(ForwardDiff.value(x))

"""
    _bessel_k_ratios_flag(z)

Core of [`bessel_k_ratios`](@ref) for a plain (non-dual) argument. Also returns
whether the regular branch was taken, which the dual method needs.

The modified Bessel functions are formed the way GSL's `gsl_sf_bessel_Kn` forms
them, as `(scaled K_n) * exp(-z)`, so that they underflow to zero at the same
argument (`z ≈ 742.7`, i.e. `θe ≈ 1.35e-3`) and the `k2 > 0` test below switches
branch where ipole's does.
"""
@inline function _bessel_k_ratios_flag(z)
    k0x = Bessels.besselk0x(z)
    k1x = Bessels.besselk1x(z)
    # Upward recurrence K_2 = K_0 + (2/z) K_1; it is stable for K_n.
    k2x = k0x + (2.0 / z) * k1x
    e = exp(-z)
    k0 = k0x * e
    k1 = k1x * e
    k2 = k2x * e
    if k2 > 0
        return k0 / k2, k1 / k2, true
    else
        # ipole: `k_ratio = (k2 > 0) ? k/k2 : 1`. The exact ratios tend to 1 as z -> inf.
        return one(z), one(z), false
    end
end

"""
    bessel_k_ratios(z)

Ratios of modified Bessel functions of the second kind, `K_0(z)/K_2(z)` and
`K_1(z)/K_2(z)`, as ipole evaluates them for the thermal rotativities (with
`z = 1/θe`), including ipole's fallback to `1` once `K_2(z)` underflows.

A method for `ForwardDiff.Dual` is provided. It does not differentiate through
the underflowing `exp(-z)` (the quotient rule would give `Inf - Inf`); it uses
the closed-form derivatives of the ratios instead,

    d(K0/K2)/dz = -K1/K2 + (K0/K2)(K1/K2) + 2 (K0/K2)/z
    d(K1/K2)/dz = -K0/K2 + (K1/K2)^2 + (K1/K2)/z

and zero in the fallback branch, where the value is the constant `1`.

# Arguments
- `z`: Argument of the Bessel functions (`1/θe`), `z > 0`.

# Returns
- A tuple `(K0/K2, K1/K2)`.
"""
@inline function bessel_k_ratios(z::Real)
    r0, r1, _ = _bessel_k_ratios_flag(z)
    return r0, r1
end

@inline function bessel_k_ratios(z::ForwardDiff.Dual{Tag}) where {Tag}
    zv = ForwardDiff.value(z)
    # Recurse on the value, so that nested duals are handled too.
    r0, r1 = bessel_k_ratios(zv)
    _, _, regular = _bessel_k_ratios_flag(_primal(zv))
    dz = ForwardDiff.partials(z)
    if regular
        dr0 = -r1 + r0 * r1 + 2.0 * r0 / zv
        dr1 = -r0 + r1 * r1 + r1 / zv
        return ForwardDiff.Dual{Tag}(r0, dr0 * dz), ForwardDiff.Dual{Tag}(r1, dr1 * dz)
    else
        return ForwardDiff.Dual{Tag}(r0, zero(r0) * dz), ForwardDiff.Dual{Tag}(r1, zero(r1) * dz)
    end
end

"""
    _faraday_x(θe, B, θ, ν)

Argument `x` of the Faraday fitting functions `f(X)` and `g(X)` of Shcherbakov
(2008) / Dexter (2016), together with the cyclotron angular frequency `omega0`.

`x` vanishes when `B == 0` or `sin(θ) == 0`. The square root is skipped there:
its value would be zero anyway, but its derivative is infinite and would poison
dual numbers with `NaN`.
"""
@inline function _faraday_x(θe, B, θ, ν)
    omega0 = Constants.EE * B / (Constants.ME * Constants.CL)
    arg = sqrt(2.0) * sin(θ) * (1.e3 * omega0 / (2.0 * π * ν))
    x = arg > 0 ? θe * sqrt(arg) : zero(θe * arg)
    return x, omega0
end

"""
    maxwell_juettner_rho_q(Ne, ν, θe, B, θ)

Faraday conversion coefficient `rho_Q` for a thermal distribution, from Dexter
(2016), eqs. B4, B6, B8 and B13 (ipole's `maxwell_juettner_rho_Q`).

# Arguments
- `Ne`: Electron number density [cm^-3].
- `ν`: Frequency in the fluid frame [Hz].
- `θe`: Dimensionless electron temperature.
- `B`: Magnetic field strength [G].
- `θ`: Angle between the wavevector and the magnetic field [rad].

# Returns
- `rho_Q` in the fluid frame [cm^-1].
"""
@inline function maxwell_juettner_rho_q(Ne, ν, θe, B, θ)
    wp2 = 4.0 * π * Ne * Constants.EE^2 / Constants.ME

    # Argument of the function f(X) (called jffunc) below.
    x, omega0 = _faraday_x(θe, B, θ, ν)

    # Modified factor f(X) of Dexter (2016). For x == 0 ipole evaluates log(0) = -Inf
    # and relies on tanh(-Inf) = -1 to switch the extra term off. Clamping the
    # argument of the log to the smallest normal number gives the same value (tanh
    # is exactly -1 there) while keeping the derivative finite.
    extraterm = (0.011 * exp(-x / 47.2) - 2.0^(-1.0 / 3.0) / 3.0^(23.0 / 6.0)
                 * π * 1.e4 * (x + 1.e-16)^(-8.0 / 3.0)) *
                (0.5 + 0.5 * tanh((log(max(x, floatmin(Float64))) - log(120.0)) / 0.1))

    jffunc = 2.011 * exp(-x^1.035 / 4.7) - cos(x / 2.0) *
             exp(-x^1.2 / 2.73) - 0.011 * exp(-x / 47.2) + extraterm

    _, k_ratio = bessel_k_ratios(1.0 / θe)

    eps11m22 = jffunc * wp2 * omega0^2 / (2.0 * π * ν)^4 *
               (k_ratio + 6.0 * θe) * sin(θ)^2

    return 2.0 * π * ν / (2.0 * Constants.CL) * eps11m22
end

"""
    maxwell_juettner_rho_v(Ne, ν, θe, B, θ)

Faraday rotation coefficient `rho_V` for a thermal distribution, using the
Shcherbakov (2008) fit. This is the branch of ipole's `maxwell_juettner_rho_V`
taken when `dexter_fit == 0`, which is what `jar_calc_dist` selects for every
regular (not field-aligned) evaluation, because the Dexter form is unstable at
low temperature.

# Arguments
- `Ne`: Electron number density [cm^-3].
- `ν`: Frequency in the fluid frame [Hz].
- `θe`: Dimensionless electron temperature.
- `B`: Magnetic field strength [G].
- `θ`: Angle between the wavevector and the magnetic field [rad].

# Returns
- `rho_V` in the fluid frame [cm^-1].
"""
@inline function maxwell_juettner_rho_v(Ne, ν, θe, B, θ)
    wp2 = 4.0 * π * Ne * Constants.EE^2 / Constants.ME

    # Argument of the function g(X) (called shgmfunc) below.
    x, omega0 = _faraday_x(θe, B, θ, ν)

    # Shcherbakov fit. Good to the smallest θe at high frequency.
    shgmfunc = 1 - 0.11 * log(1 + 0.035 * x)
    k_ratio, _ = bessel_k_ratios(1.0 / θe)
    fit_factor = k_ratio * shgmfunc

    eps12 = wp2 * omega0 / (2.0 * π * ν)^3 * fit_factor * cos(θ)

    return 2.0 * π * ν / Constants.CL * eps12
end

"""
    maxwell_juettner_rho_v_dexter(Ne, ν, θe, B, θ)

Faraday rotation coefficient `rho_V` with the Dexter (2016) modified difference
factor, eqs. B7, B8, B14 and B15. This is the branch of ipole's
`maxwell_juettner_rho_V` taken when `dexter_fit != 0`.

ipole reaches this branch only for a wavevector exactly along the magnetic field
(`θ <= 0` or `θ >= π`), where `jar_calc_dist` returns early with `dexter_fit`
still set. It is kept for that case only: it forms `(K0 - g)/K2` from the
unscaled Bessel functions, which loses all accuracy at low temperature. It is
meant to be called with plain numbers (see `Polarization.thermal_jar`).

# Arguments
- `Ne`: Electron number density [cm^-3].
- `ν`: Frequency in the fluid frame [Hz].
- `θe`: Dimensionless electron temperature.
- `B`: Magnetic field strength [G].
- `θ`: Angle between the wavevector and the magnetic field [rad].

# Returns
- `rho_V` in the fluid frame [cm^-1].
"""
@inline function maxwell_juettner_rho_v_dexter(Ne, ν, θe, B, θ)
    wp2 = 4.0 * π * Ne * Constants.EE^2 / Constants.ME
    x, omega0 = _faraday_x(θe, B, θ, ν)

    # Unscaled Bessel functions, formed as GSL does (see `_bessel_k_ratios_flag`).
    z = 1.0 / θe
    k0x = Bessels.besselk0x(z)
    k2x = k0x + (2.0 / z) * Bessels.besselk1x(z)
    e = exp(-z)
    k0 = k0x * e
    k2 = k2x * e

    if k2 > 0
        # Dexter (2016) fit using the modified difference factor g(X).
        shgmfunc = 0.43793091 * log(1.0 + 0.00185777 * x^1.50316886)
        fit_factor = (k0 - shgmfunc) / k2
    else
        # K2 underflowed: ipole falls through to the Shcherbakov form with k_ratio = 1.
        fit_factor = 1 - 0.11 * log(1 + 0.035 * x)
    end

    eps12 = wp2 * omega0 / (2.0 * π * ν)^3 * fit_factor * cos(θ)

    return 2.0 * π * ν / Constants.CL * eps12
end

end
