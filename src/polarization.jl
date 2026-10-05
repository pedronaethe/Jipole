"""
Polarized radiative transfer: evolution of the coherency tensor `N^{αβ}` along a
geodesic, and its projection onto the Stokes parameters `I, Q, U, V`.

This is a translation of ipole's `src/ipolarray.c` (transport scheme of
Mościbrodzka & Gammie 2018, MNRAS 475, 43) and of the thermal branch of
`src/model_radiation.c`. It sits on top of the existing Stokes-I code and reuses
its geodesics, tetrads, metric and Planck-function routines unchanged.

| here                              | ipole                                              |
|-----------------------------------|----------------------------------------------------|
| `stokes_to_tensor`, `tensor_to_stokes` | same names (`ipolarray.c`)                    |
| `complex_coord_to_tetrad_rank2`, `complex_tetrad_to_coord_rank2` | same names        |
| `push_polar`                      | `push_polar`                                       |
| `rotate_stokes`, `absorb_emit_stokes`, `evolve_stokes` | the Stokes-space part of `evolve_N` |
| `thermal_jar`                     | `jar_calc_dist` (`model_radiation.c`), thermal case |
| `evolve_n`                        | `evolve_N`                                         |
| `project_n`                       | `project_N`                                        |
| `polarized_step`                  | body of the loop in `integrate_emission`           |
| `integrate_emission_pol`          | `integrate_emission` + `project_N` (`get_pixel`)   |
| `save_pixel!`                     | `save_pixel` (`main.c`)                            |

# Conventions (differences from ipole are only in indexing)

- Indices are 1-based. Tetrad leg 1 is along the fluid four-velocity, leg 3 along
  the magnetic field, leg 4 along the wavevector and leg 2 completes the
  right-handed set (ipole's legs 0, 2, 3, 1). The Stokes parameters live in the
  2-3 block of the tetrad-frame tensor:
  `N[2,2] = I + Q`, `N[2,3] = U - iV`, `N[3,2] = U + iV`, `N[3,3] = I - Q`.
- `Econ[k, μ]`/`Ecov[k, μ]`: first index is the tetrad leg, second the coordinate
  index, as returned by `Tetrads.make_plasma_tetrad`.
- Transfer coefficients are the invariant combinations `j/ν²`, `α ν`, `ρ ν`, and
  the Stokes parameters are the invariant `S/ν³`. Step lengths `dlam` are the
  ones stored in the trajectory (`dl · L_unit · h/(m_e c²)`, in cm s).

# Design constraints

Every per-step routine is a pure function of immutable values (`SVector`,
`SMatrix`, tuples): it does not allocate, does not print or throw on its code
path, and contains no dynamic dispatch. This is what lets the same code run
inside the CPU pixel loop, inside a GPU kernel, and under `ForwardDiff`.

Branches are decided on `primal` values (see `MaxwellJuettnerPol.primal`), so
that a run on dual numbers follows the same code path as a run on plain numbers.
"""
module Polarization

using StaticArrays
using ..Constants
using ..GeoTypes
using ..AbstractModels
using ..Metrics
using ..Tetrads
using ..Radiation
using ..Geodesics
using ..MaxwellJuettnerPol

export PolCoeffs, PolSink, NIMG, zero_tensor, stokes_to_tensor, tensor_to_stokes,
    complex_coord_to_tetrad_rank2, complex_tetrad_to_coord_rank2, push_polar,
    rotate_stokes, absorb_emit_stokes, evolve_stokes, thermal_jar, get_pol_state,
    apply_boundary, evolve_n, project_n, polarized_step, integrate_emission_pol, save_pixel!

"""Number of images stored per pixel: Stokes I, Q, U, V and the Faraday depth (ipole's `NIMG`)."""
const NIMG = 5

"""Below this optical depth `1 - exp(-τ)` is replaced by its Taylor series (ipole's `CUT_SMALL_OPTICAL_DEPTH`)."""
const CUT_SMALL_OPTICAL_DEPTH = 1e-5

"""Smallest squared coefficient norm that can be safely divided by (ipole's `CUT_PREVENT_NAN`)."""
const CUT_PREVENT_NAN = 1e-80

"""Largest fractional polarization allowed for the emissivity (ipole's `max_pol_frac_e`)."""
const MAX_POL_FRAC_E = 0.99

"""
    PolCoeffs{T}

The eleven invariant polarized transfer coefficients at one point, in the plasma
tetrad: emissivities `jI, jQ, jU, jV` (`j/ν²`), absorptivities `aI, aQ, aU, aV`
(`α ν`) and rotativities `rQ, rU, rV` (`ρ ν`).
"""
struct PolCoeffs{T}
    jI::T
    jQ::T
    jU::T
    jV::T
    aI::T
    aQ::T
    aU::T
    aV::T
    rQ::T
    rU::T
    rV::T
end

Base.zero(::Type{PolCoeffs{T}}) where {T} = PolCoeffs{T}(zero(T), zero(T), zero(T), zero(T), zero(T),
    zero(T), zero(T), zero(T), zero(T), zero(T), zero(T))

# ---------------------------------------------------------------------------
# Coherency tensor <-> Stokes parameters
# ---------------------------------------------------------------------------

"""
    zero_tensor(T)

The coherency tensor `N^{αβ}` of a ray carrying no radiation: a 4×4 complex
matrix of zeros with real type `T`.
"""
@inline zero_tensor(::Type{T}) where {T} = zero(SMatrix{4,4,Complex{T},16})

"""
    stokes_to_tensor(N_tetrad, SI, SQ, SU, SV)

Write the Stokes parameters into the tetrad-frame coherency tensor (ipole's
`stokes_to_tensor`). Only the four components of the 2-3 block are replaced;
the other twelve are carried through unchanged, as in ipole.

# Arguments
- `N_tetrad`: Tetrad-frame coherency tensor to update.
- `SI`, `SQ`, `SU`, `SV`: Stokes parameters.

# Returns
- The updated tetrad-frame tensor.
"""
@inline function stokes_to_tensor(N_tetrad::SMatrix{4,4,Complex{TN},16}, SI, SQ, SU, SV) where {TN}
    T = promote_type(TN, typeof(SI), typeof(SQ), typeof(SU), typeof(SV))
    z = zero(T)
    # Column-major: each line below is one column of the matrix.
    return SMatrix{4,4,Complex{T},16}(
        N_tetrad[1, 1], N_tetrad[2, 1], N_tetrad[3, 1], N_tetrad[4, 1],
        N_tetrad[1, 2], complex(SI + SQ, z), complex(SU, SV), N_tetrad[4, 2],
        N_tetrad[1, 3], complex(SU, -SV), complex(SI - SQ, z), N_tetrad[4, 3],
        N_tetrad[1, 4], N_tetrad[2, 4], N_tetrad[3, 4], N_tetrad[4, 4]
    )
end

"""
    tensor_to_stokes(N_tetrad)

Read the Stokes parameters off the tetrad-frame coherency tensor (ipole's
`tensor_to_stokes`).

# Arguments
- `N_tetrad`: Tetrad-frame coherency tensor.

# Returns
- A tuple `(SI, SQ, SU, SV)`.
"""
@inline function tensor_to_stokes(N_tetrad)
    SI = real(N_tetrad[2, 2] + N_tetrad[3, 3]) / 2
    SQ = real(N_tetrad[2, 2] - N_tetrad[3, 3]) / 2
    SU = real(N_tetrad[2, 3] + N_tetrad[3, 2]) / 2
    SV = imag(N_tetrad[3, 2] - N_tetrad[2, 3]) / 2
    return SI, SQ, SU, SV
end

"""
    complex_coord_to_tetrad_rank2(N_coord, Ecov)

Project a contravariant rank-2 tensor from the coordinate basis onto a tetrad:
`N^{(a)(b)} = Ecov[a, μ] Ecov[b, ν] N^{μν}` (ipole's
`complex_coord_to_tetrad_rank2`).

# Arguments
- `N_coord`: Tensor components in the coordinate basis.
- `Ecov`: Covariant tetrad, as returned by `Tetrads.make_plasma_tetrad`.

# Returns
- The tensor components in the tetrad basis.
"""
@inline complex_coord_to_tetrad_rank2(N_coord, Ecov) = Ecov * N_coord * transpose(Ecov)

"""
    complex_tetrad_to_coord_rank2(N_tetrad, Econ)

Inverse of [`complex_coord_to_tetrad_rank2`](@ref):
`N^{μν} = Econ[a, μ] Econ[b, ν] N^{(a)(b)}` (ipole's
`complex_tetrad_to_coord_rank2`).

# Arguments
- `N_tetrad`: Tensor components in the tetrad basis.
- `Econ`: Contravariant tetrad, as returned by `Tetrads.make_plasma_tetrad`.

# Returns
- The tensor components in the coordinate basis.
"""
@inline complex_tetrad_to_coord_rank2(N_tetrad, Econ) = transpose(Econ) * N_tetrad * Econ

# ---------------------------------------------------------------------------
# Parallel transport
# ---------------------------------------------------------------------------

"""
    push_polar(Ni, Nm, lconn, Km, dl)

Parallel-transport the coherency tensor over an affine step `dl` (ipole's
`push_polar`):

    Nf^{ij} = Ni^{ij} - (Γ^i_{kl} Nm^{kj} + Γ^j_{kl} Nm^{ik}) Km^l dl

Called twice per step it forms a second-order (midpoint) scheme; see
[`polarized_step`](@ref).

# Arguments
- `Ni`: Tensor at the start of the step.
- `Nm`: Tensor used to evaluate the right-hand side.
- `lconn`: Connection coefficients `Γ^μ_{αβ}` at the evaluation point.
- `Km`: Wavevector at the evaluation point.
- `dl`: Affine step length, in code units.

# Returns
- The tensor at the end of the step.
"""
@inline function push_polar(Ni, Nm, lconn, Km, dl)
    # A^i_k = Γ^i_{kl} K^l, so that dN/dλ = -(A N + N Aᵀ).
    A = @SMatrix [lconn[i, k, 1] * Km[1] + lconn[i, k, 2] * Km[2] +
                  lconn[i, k, 3] * Km[3] + lconn[i, k, 4] * Km[4] for i in 1:4, k in 1:4]
    return Ni - (A * Nm + Nm * transpose(A)) * dl
end

# ---------------------------------------------------------------------------
# Constant-coefficient solution of the transfer equation in Stokes space
# ---------------------------------------------------------------------------

"""
    rotate_stokes(SQ, SU, SV, rQ, rU, rV, x)

Exact solution of the Faraday rotation/conversion part of the transfer
equation, `dS/dλ = ρ × S` for `S = (Q, U, V)`, over a path `x` with constant
`ρ = (rQ, rU, rV)`: a rotation of `S` about `ρ` by the angle `|ρ| x`
(Mościbrodzka & Gammie 2018, appendix A1). Stokes I is unaffected.

# Arguments
- `SQ`, `SU`, `SV`: Stokes parameters at the start.
- `rQ`, `rU`, `rV`: Invariant rotativities.
- `x`: Path length (`dlam`, or `dlam/2` for a half step).

# Returns
- A tuple `(SQ, SU, SV)` at the end.
"""
@inline function rotate_stokes(SQ, SU, SV, rQ, rU, rV, x)
    rho2 = rQ * rQ + rU * rU + rV * rV
    if primal(rho2) > CUT_PREVENT_NAN
        rho = sqrt(rho2)
        rdS = rQ * SQ + rU * SU + rV * SV
        c = cos(rho * x)
        s = sin(rho * x)
        sh = sin(0.5 * rho * x)
        Q = SQ * c + 2 * rQ * rdS / rho2 * sh * sh + (rU * SV - rV * SU) / rho * s
        U = SU * c + 2 * rU * rdS / rho2 * sh * sh + (rV * SQ - rQ * SV) / rho * s
        V = SV * c + 2 * rV * rdS / rho2 * sh * sh + (rQ * SU - rU * SQ) / rho * s
        return Q, U, V
    else
        # Negligible rotativity: first-order form, free of the division by |ρ|.
        Q = SQ + (-rV * SU + rU * SV) * x
        U = SU + (rV * SQ - rQ * SV) * x
        V = SV + (-rU * SQ + rQ * SU) * x
        return Q, U, V
    end
end

"""
    absorb_emit_stokes(SI, SQ, SU, SV, c, x)

Exact solution of the emission/absorption part of the transfer equation (no
Faraday rotation) over a path `x` with constant coefficients. The absorption
matrix has eigenvalues `aI` (twice), `aI + aP` and `aI - aP`, with
`aP = sqrt(aQ² + aU² + aV²)` (Mościbrodzka & Gammie 2018, appendix A2).

The factors `1 - exp(-τ)` are replaced by their Taylor series at small optical
depth to avoid loss of precision, with ipole's nested conditions. When `aP`
vanishes the four Stokes parameters decouple and each is advanced by
`Radiation.approximate_solve`.

The expressions assume `aP < aI`, which `thermal_jar` guarantees by capping the
fractional polarization of the emissivity.

# Arguments
- `SI`, `SQ`, `SU`, `SV`: Stokes parameters at the start.
- `c`: Transfer coefficients ([`PolCoeffs`](@ref)).
- `x`: Path length (`dlam`).

# Returns
- A tuple `(SI, SQ, SU, SV)` at the end.
"""
@inline function absorb_emit_stokes(SI1, SQ1, SU1, SV1, c::PolCoeffs, x)
    jI, jQ, jU, jV = c.jI, c.jQ, c.jU, c.jV
    aI, aQ, aU, aV = c.aI, c.aQ, c.aU, c.aV

    aP2 = aQ * aQ + aU * aU + aV * aV
    if primal(aP2) > CUT_PREVENT_NAN
        aP = sqrt(aP2)
        tauP = aP * x
        tauI = aI * x
        ads0 = aQ * SQ1 + aU * SU1 + aV * SV1
        adj = aQ * jQ + aU * jU + aV * jV
        efacm = exp(-tauI + tauP)
        efacp = exp(-tauI - tauP)
        efac = exp(-tauI)

        # Effect of absorption on the initial Stokes vector; safe for all optical depths.
        SI2 = efacm * (SI1 / 2 - ads0 / (2 * aP)) +
              efacp * (SI1 / 2 + ads0 / (2 * aP))

        SQ2 = efacm * (-SI1 * aQ * aP + ads0 * aQ) / (2 * aP2) +
              efacp * (SI1 * aQ * aP + ads0 * aQ) / (2 * aP2) +
              efac * (SQ1 - ads0 * aQ / aP2)

        SU2 = efacm * (-SI1 * aU * aP + ads0 * aU) / (2 * aP2) +
              efacp * (SI1 * aU * aP + ads0 * aU) / (2 * aP2) +
              efac * (SU1 - ads0 * aU / aP2)

        SV2 = efacm * (-SI1 * aV * aP + ads0 * aV) / (2 * aP2) +
              efacp * (SI1 * aV * aP + ads0 * aV) / (2 * aP2) +
              efac * (SV1 - ads0 * aV / aP2)

        # Taylor series at small optical depth. The conditions are nested as in ipole.
        afacm = 1 - efacm
        afac = 1 - efac
        afacp = 1 - efacp
        if primal(tauI - tauP) <= CUT_SMALL_OPTICAL_DEPTH
            e = tauI - tauP
            afacm = e * (1 - (e / 2) * (1 - e / 3))

            if primal(tauI) <= CUT_SMALL_OPTICAL_DEPTH
                e = tauI
                afac = e * (1 - (e / 2) * (1 - e / 3))

                if primal(tauI + tauP) <= CUT_SMALL_OPTICAL_DEPTH
                    e = tauI + tauP
                    afacp = e * (1 - (e / 2) * (1 - e / 3))
                end
            end
        end

        # Emission: piece proportional to afac ...
        SQ2 += afac * (jQ / aI - aQ * adj / (aI * aP2))
        SU2 += afac * (jU / aI - aU * adj / (aI * aP2))
        SV2 += afac * (jV / aI - aV * adj / (aI * aP2))

        # ... pieces proportional to afacm ...
        SI2 += afacm * (aP * jI - adj) / (2 * aP * (aI - aP))
        SQ2 += afacm * aQ * (-aP * jI + adj) / (2 * aP2 * (aI - aP))
        SU2 += afacm * aU * (-aP * jI + adj) / (2 * aP2 * (aI - aP))
        SV2 += afacm * aV * (-aP * jI + adj) / (2 * aP2 * (aI - aP))

        # ... and pieces proportional to afacp.
        SI2 += afacp * (aP * jI + adj) / (2 * aP * (aI + aP))
        SQ2 += afacp * aQ * (aP * jI + adj) / (2 * aP2 * (aI + aP))
        SU2 += afacp * aU * (aP * jI + adj) / (2 * aP2 * (aI + aP))
        SV2 += afacp * aV * (aP * jI + adj) / (2 * aP2 * (aI + aP))

        return SI2, SQ2, SU2, SV2
    else
        # aP == 0: unpolarized absorption, the four equations are independent.
        SI2 = Radiation.approximate_solve(SI1, jI, aI, jI, aI, x)
        SQ2 = Radiation.approximate_solve(SQ1, jQ, aI, jQ, aI, x)
        SU2 = Radiation.approximate_solve(SU1, jU, aI, jU, aI, x)
        SV2 = Radiation.approximate_solve(SV1, jV, aI, jV, aI, x)
        return SI2, SQ2, SU2, SV2
    end
end

"""
    evolve_stokes(SI, SQ, SU, SV, c, dlam)

Advance the Stokes parameters over one step with constant transfer
coefficients, by operator splitting: Faraday rotation over `dlam/2`,
emission and absorption over `dlam`, Faraday rotation over `dlam/2` (the
Stokes-space part of ipole's `evolve_N`). The splitting is second-order
accurate in `dlam`, and exact when either the rotativities or the
emission/absorption coefficients vanish.

# Arguments
- `SI`, `SQ`, `SU`, `SV`: Invariant Stokes parameters at the start of the step.
- `c`: Transfer coefficients ([`PolCoeffs`](@ref)).
- `dlam`: Step length.

# Returns
- A tuple `(SI, SQ, SU, SV)` at the end of the step.
"""
@inline function evolve_stokes(SI0, SQ0, SU0, SV0, c::PolCoeffs, dlam)
    SQ1, SU1, SV1 = rotate_stokes(SQ0, SU0, SV0, c.rQ, c.rU, c.rV, dlam * 0.5)
    SI2, SQ2, SU2, SV2 = absorb_emit_stokes(SI0, SQ1, SU1, SV1, c, dlam)
    SQ, SU, SV = rotate_stokes(SQ2, SU2, SV2, c.rQ, c.rU, c.rV, dlam * 0.5)
    return SI2, SQ, SU, SV
end

# ---------------------------------------------------------------------------
# Transfer coefficients for a thermal plasma
# ---------------------------------------------------------------------------

"""
    thermal_jar(Ne, ν, θe, B, θ)

Invariant polarized transfer coefficients of a thermal plasma, in the plasma
tetrad. This is the thermal branch (`emission_type = 4`, ipole's default) of
`jar_calc_dist` in ipole's `model_radiation.c`:

- emissivities from the Dexter (2016) fits, with `jQ` sign-flipped from the fit
  convention to the tetrad convention and `jU = 0`;
- polarized emissivity capped at `MAX_POL_FRAC_E` of `jI`, since the transfer
  solution breaks down at 100% polarization;
- absorptivities from Kirchhoff's law, `a_S = j_S / B_ν`;
- `rQ` from Dexter (2016), `rV` from Shcherbakov (2008), `rU = 0`.

For a wavevector exactly along the field (`θ <= 0` or `θ >= π`) there is no
emission or absorption and only Faraday rotation survives. That branch is
evaluated on plain numbers (it is a set of measure zero, and the angle has an
infinite derivative there), so it contributes no derivative under AD.

# Arguments
- `Ne`: Electron number density [cm^-3].
- `ν`: Frequency in the fluid frame [Hz].
- `θe`: Dimensionless electron temperature.
- `B`: Magnetic field strength [G].
- `θ`: Angle between the wavevector and the magnetic field [rad].

# Returns
- The coefficients as a [`PolCoeffs`](@ref).
"""
@inline function thermal_jar(Ne, ν, θe, B, θ)
    T = promote_type(typeof(Ne), typeof(ν), typeof(θe), typeof(B), typeof(θ))
    z = zero(T)

    # Don't emit where there are no electrons. A non-positive frequency cannot occur
    # for a future-directed photon; it is excluded so the fits never see it.
    if !(primal(Ne) > 0) || !(primal(ν) > 0)
        return zero(PolCoeffs{T})
    end

    # No emission/absorption along field lines, but keep Faraday rotation.
    if primal(θ) <= 0 || primal(θ) >= Float64(π)
        rV = MaxwellJuettnerPol.maxwell_juettner_rho_v_dexter(primal(Ne), primal(ν), primal(θe), primal(B), primal(θ)) * primal(ν)
        return PolCoeffs{T}(z, z, z, z, z, z, z, z, z, z, T(rV))
    end

    nusq = ν * ν

    # EMISSIVITIES, made invariant by the division by ν².
    jI_fit, jQ_fit, jV_fit = MaxwellJuettnerPol.maxwell_juettner_dexter_iqv(Ne, ν, θe, B, θ)
    jI = jI_fit / nusq
    jQ = -jQ_fit / nusq
    jU = z
    jV = jV_fit / nusq

    # Transport does not like 100% polarization.
    jP2 = jQ * jQ + jU * jU + jV * jV
    jP = primal(jP2) > 0 ? sqrt(jP2) : zero(jP2)
    if primal(jI) < primal(jP) / MAX_POL_FRAC_E
        pol_frac_e = jI / jP * MAX_POL_FRAC_E
        jQ *= pol_frac_e
        jU *= pol_frac_e
        jV *= pol_frac_e
    end

    # ABSORPTIVITIES from Kirchhoff's law. Already invariant, and aI > aP by construction.
    Bnuinv = Radiation.bnu_inv(ν, θe)
    if primal(Bnuinv) > 0
        aI = jI / Bnuinv
        aQ = jQ / Bnuinv
        aU = jU / Bnuinv
        aV = jV / Bnuinv
    else
        aI = z
        aQ = z
        aU = z
        aV = z
    end

    # ROTATIVITIES, made invariant by the multiplication by ν.
    rQ = MaxwellJuettnerPol.maxwell_juettner_rho_q(Ne, ν, θe, B, θ) * ν
    rU = z
    rV = MaxwellJuettnerPol.maxwell_juettner_rho_v(Ne, ν, θe, B, θ) * ν

    return PolCoeffs{T}(jI, jQ, jU, jV, aI, aQ, aU, aV, rQ, rU, rV)
end

# ---------------------------------------------------------------------------
# Model interface
# ---------------------------------------------------------------------------

"""
    get_pol_state(X, Kcon, freq, bhspin, model, data)

Model hook: everything the polarized source step needs at the point `X`. Each
model that emits along the ray implements a method (see
`models/iharm_pol.jl`), in the same way models implement `Radiation.get_jk`
for Stokes I.

The four-velocity and field are returned together with the coefficients so
that they are evaluated once per step. They must be returned even where the
plasma does not emit (`Ne <= 0`), because the tetrad is still needed there.

# Arguments
- `X`: Position four-vector in internal coordinates.
- `Kcon`: Contravariant photon 4-momentum.
- `freq`: Observing frequency [Hz].
- `bhspin`: Dimensionless black hole spin parameter.
- `model`: Model parameters.
- `data`: Model data (GRMHD snapshot(s), or `nothing`).

# Returns
- A tuple `(coeffs, Ucon, Bcon, Bcov)`: the [`PolCoeffs`](@ref), the fluid
  four-velocity and the magnetic field four-vector (contravariant and covariant).
"""
function get_pol_state end

"""
    apply_boundary(N, Xi, Xf, Kconf, bhspin, model)

Model hook: replace the radiation carried by the ray when the step from `Xi` to
`Xf` crosses an emitting surface. The default does nothing; the thin disk
implements it (ipole's `#if THIN_DISK` block in `integrate_emission`).

# Arguments
- `N`: Coherency tensor in the coordinate basis, already transported to `Xf`.
- `Xi`, `Xf`: Start and end of the step, in internal coordinates.
- `Kconf`: Photon 4-momentum at `Xf`.
- `bhspin`: Dimensionless black hole spin parameter.
- `model`: Model parameters.

# Returns
- The (possibly replaced) coherency tensor.
"""
@inline apply_boundary(N, Xi, Xf, Kconf, bhspin, model::AbstractModel) = N

# ---------------------------------------------------------------------------
# One step of polarized transfer
# ---------------------------------------------------------------------------

"""
    plasma_tetrad(Ucon, Kcon, Bcon, gcov)

`Tetrads.make_plasma_tetrad` with its four inputs first promoted to a common
element type. The existing routine takes its type from `Ucon` alone, which
fails when only some of the inputs carry dual numbers.

# Returns
- A tuple `(flag, Econ, Ecov)`, as `Tetrads.make_plasma_tetrad`.
"""
@inline function plasma_tetrad(Ucon, Kcon, Bcon, gcov)
    T = promote_type(eltype(Ucon), eltype(Kcon), eltype(Bcon), eltype(gcov))
    return Tetrads.make_plasma_tetrad(SVector{4,T}(Ucon), SVector{4,T}(Kcon), SVector{4,T}(Bcon), SMatrix{4,4,T,16}(gcov))
end

"""
    evolve_n(N, tauF, Xf, Kconf, dlam, freq, bhspin, model, data)

Apply emission, absorption and Faraday rotation to the coherency tensor over
one step (ipole's `evolve_N`). The transfer coefficients are evaluated at the
end point `Xf` and held constant over the step:

1. build the plasma tetrad at `Xf` (leg 3 along the field, leg 4 along the ray);
2. project `N` onto it and read off the Stokes parameters;
3. advance them with [`evolve_stokes`](@ref);
4. write them back and return to the coordinate basis.

# Arguments
- `N`: Coherency tensor in the coordinate basis, already transported to `Xf`.
- `tauF`: Faraday depth accumulated so far.
- `Xf`, `Kconf`: Position and photon 4-momentum at the end of the step.
- `dlam`: Step length as stored in the trajectory.
- `freq`: Observing frequency [Hz].
- `bhspin`: Dimensionless black hole spin parameter.
- `model`, `data`: Model parameters and data.

# Returns
- A tuple `(N, tauF)`.
"""
@inline function evolve_n(N, tauF, Xf, Kconf, dlam, freq, bhspin, model, data)
    coeffs, Ucon, Bcon, Bcov = get_pol_state(Xf, Kconf, freq, bhspin, model, data)

    # Guess B if we absolutely must (e.g. outside the simulation domain), so that
    # the tetrad is still defined.
    bsq = Bcon[1] * Bcov[1] + Bcon[2] * Bcov[2] + Bcon[3] * Bcov[3] + Bcon[4] * Bcov[4]
    if primal(bsq) <= 0
        o = one(eltype(Bcon))
        Bcon = typeof(Bcon)(zero(o), o, o, o)
    end

    gcov = Metrics.gcov_func(Xf, bhspin, model)
    _, Econ, Ecov = plasma_tetrad(Ucon, Kconf, Bcon, gcov)

    # Convert N to Stokes parameters in the plasma frame, evolve them, convert back.
    N_tetrad = complex_coord_to_tetrad_rank2(N, Ecov)
    SI0, SQ0, SU0, SV0 = tensor_to_stokes(N_tetrad)
    SI, SQ, SU, SV = evolve_stokes(SI0, SQ0, SU0, SV0, coeffs, dlam)

    tauF += dlam * abs(coeffs.rV)

    N_tetrad = stokes_to_tensor(N_tetrad, SI, SQ, SU, SV)
    return complex_tetrad_to_coord_rank2(N_tetrad, Econ), tauF
end

"""
    polarized_step(N, tauF, Xi, Kconi, Xhalf, Kconhalf, Xf, Kconf, dlam, dl_unit, Rh, freq, bhspin, model, data)

Advance the coherency tensor over one geodesic step, from the far point `Xi`
to the near point `Xf` (toward the camera). This is the body of the loop in
ipole's `integrate_emission`:

1. parallel transport, with a midpoint scheme: a half step using the
   connection and wavevector at `Xi`, then the full step using those at the
   half-step point `Xhalf`;
2. the model's boundary condition, if the step crosses an emitting surface;
3. emission, absorption and Faraday rotation ([`evolve_n`](@ref)), if `Xf` is
   inside the radiating region.

It takes plain values rather than a trajectory, so that the CPU loop
([`integrate_emission_pol`](@ref)) and the GPU kernel share it whatever their
trajectory storage.

# Arguments
- `N`, `tauF`: Coherency tensor and Faraday depth at `Xi`.
- `Xi`, `Kconi`: Position and photon 4-momentum at the far end of the step.
- `Xhalf`, `Kconhalf`: The same at the half-step point of the geodesic integrator.
- `Xf`, `Kconf`: The same at the near end of the step.
- `dlam`: Step length as stored in the trajectory (cm s).
- `dl_unit`: Conversion `L_unit · h/(m_e c²)` between that and the affine step in code units.
- `Rh`: Event horizon radius.
- `freq`: Observing frequency [Hz].
- `bhspin`: Dimensionless black hole spin parameter.
- `model`, `data`: Model parameters and data.

# Returns
- A tuple `(N, tauF)` at `Xf`.
"""
@inline function polarized_step(N, tauF, Xi, Kconi, Xhalf, Kconhalf, Xf, Kconf, dlam, dl_unit, Rh, freq, bhspin, model, data)
    # Parallel transport. A ray that carries no radiation yet stays empty, so the
    # two connection evaluations are skipped; the result is identical.
    if !iszero(N)
        dl = dlam / dl_unit
        Nh = push_polar(N, N, Geodesics.get_connection_analytic(Xi, bhspin, model), Kconi, 0.5 * dl)
        N = push_polar(N, Nh, Geodesics.get_connection_analytic(Xhalf, bhspin, model), Kconhalf, dl)
    end

    N = apply_boundary(N, Xi, Xf, Kconf, bhspin, model)

    if Radiation.radiating_region(Xf, model, Rh)
        N, tauF = evolve_n(N, tauF, Xf, Kconf, dlam, freq, bhspin, model, data)
    end

    return N, tauF
end

# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

"""
    project_n(N, Xcam, bhspin, model, rotcam=0.0)

Project the coherency tensor onto the camera tetrad and return the Stokes
parameters measured by the camera (ipole's `project_N`).

`rotcam` rotates the Q/U axes with the camera. Jipole's camera has no rotation
yet, so it is always called with the default `0`.

# Arguments
- `N`: Coherency tensor in the coordinate basis, at the camera.
- `Xcam`: Camera position in internal coordinates.
- `bhspin`: Dimensionless black hole spin parameter.
- `model`: Model parameters.
- `rotcam`: Camera rotation angle [rad].

# Returns
- A tuple `(SI, SQ, SU, SV)` of invariant Stokes parameters.
"""
@inline function project_n(N, Xcam, bhspin, model, rotcam=0.0)
    _, _, Ecov = Tetrads.make_camera_tetrad(Xcam, bhspin, model)
    N_tetrad = complex_coord_to_tetrad_rank2(N, Ecov)
    SI, Q, U, SV = tensor_to_stokes(N_tetrad)

    rot = -2 * rotcam
    SQ = Q * cos(rot) - U * sin(rot)
    SU = Q * sin(rot) + U * cos(rot)
    return SI, SQ, SU, SV
end

# ---------------------------------------------------------------------------
# Integration along a stored trajectory
# ---------------------------------------------------------------------------

"""
    integrate_emission_pol(traj, nsteps, freq, bhspin, model, data=nothing)

Integrate the polarized transfer equation along a stored geodesic, from its far
end to the camera, and project the result onto the camera (ipole's
`integrate_emission` followed by `project_N`). This is the polarized
counterpart of `Radiation.integrate_emission!` and walks the same trajectory.

Layout of the trajectory (`traj[1]` is the camera): for the step from `traj[n]`
to `traj[n-1]`, the half-step point is stored in `traj[n]` and the step length
in `traj[n-1]`.

# Arguments
- `traj`: Geodesic trajectory, as filled by `Geodesics.trace_geodesic`.
- `nsteps`: Number of points in the trajectory.
- `freq`: Observing frequency [Hz].
- `bhspin`: Dimensionless black hole spin parameter.
- `model`, `data`: Model parameters and data.

# Returns
- A tuple `(SI, SQ, SU, SV, tauF)`: invariant Stokes parameters at the camera
  (multiply by `freq^3` for specific intensities) and the Faraday depth.
"""
function integrate_emission_pol(traj::Vector{GeoTypes.OfTrajGeneric{T}}, nsteps::Int, freq, bhspin, model::AbstractModel, data=nothing) where {T}
    Rh = 1 + sqrt(1.0 - bhspin * bhspin)
    dl_unit = model.L_unit * Constants.HPL / (Constants.ME * Constants.CL * Constants.CL)

    N = zero_tensor(T)
    tauF = zero(T)
    @inbounds for nstep = nsteps:-1:2
        ti = traj[nstep]
        tf = traj[nstep-1]
        N, tauF = polarized_step(N, tauF, ti.X, ti.Kcon, ti.Xhalf, ti.Kconhalf, tf.X, tf.Kcon,
            tf.dl, dl_unit, Rh, freq, bhspin, model, data)
    end

    SI, SQ, SU, SV = project_n(N, traj[1].X, bhspin, model)
    return SI, SQ, SU, SV, tauF
end

"""
    save_pixel!(pol, I, J, SI, SQ, SU, SV, tauF, freq, qu_conv)

Store one pixel of the polarized image (ipole's `save_pixel`): the invariant
Stokes parameters are converted to specific intensities by `freq^3`, and Q and
U are negated for `qu_conv == 0` so that the EVPA is measured East of North
(the IAU convention, ipole's default). Stokes V is never sign-changed.

# Arguments
- `pol`: Output array of size `(NIMG, nx, ny)`: I, Q, U, V and Faraday depth.
- `I`, `J`: Pixel indices.
- `SI`, `SQ`, `SU`, `SV`: Invariant Stokes parameters at the camera.
- `tauF`: Faraday depth.
- `freq`: Observing frequency [Hz].
- `qu_conv`: `0` for EVPA East of North, `1` for North of West.
"""
@inline function save_pixel!(pol, I, J, SI, SQ, SU, SV, tauF, freq, qu_conv)
    f3 = freq^3
    sgn = qu_conv == 0 ? -1 : 1
    @inbounds begin
        pol[1, I, J] = SI * f3
        pol[2, I, J] = sgn * SQ * f3
        pol[3, I, J] = sgn * SU * f3
        pol[4, I, J] = SV * f3
        pol[5, I, J] = tauF
    end
    return nothing
end

"""
    PolSink(data, pol, qu_conv=0)

Wrapper around a model's `data` that makes the existing pixel loop
(`Imaging.raytrace_image`) produce a polarized image as well. The loop passes
its `simulation_data` argument straight to `Radiation.integrate_emission!`;
handing it a `PolSink` selects the method below, which runs the unchanged
Stokes-I integrator on `data` and then the polarized integrator on the same
trajectory, writing into `pol`.

# Fields
- `data`: The model data the Stokes-I integrator expects (GRMHD snapshot(s), or `nothing`).
- `pol`: Output array of size `(NIMG, nx, ny)`.
- `qu_conv`: Q/U sign convention, see [`save_pixel!`](@ref).
"""
struct PolSink{D,A<:AbstractArray}
    data::D
    pol::A
    qu_conv::Int
end

PolSink(data, pol) = PolSink(data, pol, 0)

"""
    integrate_emission_sink!(traj, nsteps, Image, I, J, freq, bhspin, model, sink)

Compute one pixel of both images (ipole's `get_pixel` + `save_pixel`): the
unpolarized intensity `Image[I, J]`, with the model's existing
`Radiation.integrate_emission!` method, and the polarized pixel
`sink.pol[:, I, J]`, with [`integrate_emission_pol`](@ref).

As in ipole, the two are separate calculations: the unpolarized image uses the
Leung et al. (2011) emissivity averaged over each step, the polarized one the
Dexter (2016) fits evaluated at the end of each step. Their Stokes I therefore
differ slightly, by design.
"""
@inline function integrate_emission_sink!(traj, nsteps, Image, I, J, freq, bhspin, model, sink::PolSink)
    Radiation.integrate_emission!(traj, nsteps, Image, I, J, freq, bhspin, model, sink.data)
    SI, SQ, SU, SV, tauF = integrate_emission_pol(traj, nsteps, freq, bhspin, model, sink.data)
    save_pixel!(sink.pol, I, J, SI, SQ, SU, SV, tauF, freq, sink.qu_conv)
    return nothing
end

"""
    Radiation.integrate_emission!(traj, nsteps, Image, I, J, freq, bhspin, model, sink::PolSink)

Polarized pixel integration: selected when the pixel loop is given a
[`PolSink`](@ref) in place of the model data. See
[`integrate_emission_sink!`](@ref).
"""
function Radiation.integrate_emission!(traj::Vector{GeoTypes.OfTrajGeneric{T}}, nsteps::Int, Image, I, J, freq, bhspin, model::AbstractModel, sink::PolSink) where {T}
    return integrate_emission_sink!(traj, nsteps, Image, I, J, freq, bhspin, model, sink)
end

end
