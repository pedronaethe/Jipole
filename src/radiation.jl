"""
Radiative transfer along a geodesic.

`get_jk` and `radiating_region` are model-specific and declared here only
as generic function stubs; each model (`Analytic`, `ThinDisk`, `Iharm`)
provides its own method. `integrate_emission!` has a default
implementation (the volumetric `get_jk` + [`approximate_solve`](@ref)
loop, used by `Analytic`/`Iharm`), which `ThinDisk` overrides with its
boundary-crossing emission model.
"""
module Radiation

using StaticArrays
using ..Constants
using ..GeoTypes
using ..AbstractModels
using ..DebugFunctions

export get_fluid_nu, Bnu_inv, get_jk, integrate_emission!, get_bk_angle,
    approximate_solve, radiating_region

"""
    get_fluid_nu(Kcon, Ucov)

Compute the photon frequency in the fluid frame.

# Arguments
- `Kcon`: Contravariant 4-momentum of the photon, in internal coordinates.
- `Ucov`: Covariant 4-velocity of the fluid, in internal coordinates.

# Returns
- The fluid-frame frequency.
"""
function get_fluid_nu(Kcon, Ucov)
    nu = -(Kcon[1] * Ucov[1] + Kcon[2] * Ucov[2] + Kcon[3] * Ucov[3] + Kcon[4] * Ucov[4]) * Constants.ME * Constants.CL * Constants.CL / Constants.HPL
    return nu
end

"""
    Bnu_inv(nu, θe)

Compute the invariant Planck function `Bnu / nu^3`.

# Arguments
- `nu`: Frequency.
- `θe`: Dimensionless electron temperature.

# Returns
- The invariant Planck function.
"""
function Bnu_inv(nu, θe)
    x = Constants.HPL * nu / (Constants.ME * Constants.CL * Constants.CL * θe)

    if x < 2.e-3
        return ((2.0 * Constants.HPL / (Constants.CL * Constants.CL))
                /
                (x / 24.0 * (24.0 + x * (12.0 + x * (4.0 + x)))))
    else
        return ((2.0 * Constants.HPL / (Constants.CL * Constants.CL)) / (exp(x) - 1.0))
    end
end

"""
    get_jk(X, Kcon, freq, bhspin, model, data=nothing, derivative_calculation=Val(false))

Compute the (invariant) emissivity `j` and absorption coefficient `k` at
position `X`.

This is model-specific: each model provides its own method, dispatched on
`model`'s type. `derivative_calculation` additionally requests the
derivatives of `j`/`k` with respect to `Rhigh` (used by `Iharm`); models
that don't support this simply ignore it.

# Arguments
- `X`: Position four-vector in internal coordinates.
- `Kcon`: Contravariant 4-momentum of the photon, in internal coordinates.
- `freq`: Pivotal frequency, in cgs units.
- `bhspin`: Dimensionless black hole spin parameter.
- `model`: Model parameters, used to select the emission model.
- `data`: Model-specific auxiliary data (e.g. `Iharm`'s GRMHD snapshots).
- `derivative_calculation`: `Val(true)` to additionally return
  `dj/dRhigh`, `dk/dRhigh`.

# Returns
- A tuple `(j, k, dj_dRhigh, dk_dRhigh)` (the last two are zero unless
  `derivative_calculation === Val(true)` and the model supports it).
"""
function get_jk end

"""
    radiating_region(X, model, Rh)

Check whether the position `X` lies within the model's radiating region.

This is model-specific: each model provides its own method, dispatched on
`model`'s type.

# Arguments
- `X`: Position four-vector in internal coordinates.
- `model`: Model parameters.
- `Rh`: Event horizon radius.

# Returns
- `true` if `X` is within the radiating region.
"""
function radiating_region end

"""
    integrate_emission!(traj, nsteps, Image, I, J, freq, bhspin, model, data=nothing)

Integrate the emission along the geodesic trajectory `traj`, writing the
resulting intensity into `Image[I, J]`.

This default method implements the volumetric radiative transfer used by
`Analytic`/`Iharm` (accumulating `get_jk` via [`approximate_solve`](@ref)
over the whole trajectory); `ThinDisk` overrides it with a boundary-
crossing emission model.

# Arguments
- `traj`: Vector of geodesic trajectory steps.
- `nsteps`: Number of steps in the trajectory.
- `Image`: Output image, `Image[I, J]` is overwritten with the intensity.
- `I`, `J`: Pixel indices in the image plane.
- `freq`: Pivotal frequency, in cgs units.
- `bhspin`: Dimensionless black hole spin parameter.
- `model`: Model parameters.
- `data`: Model-specific auxiliary data.
"""
function integrate_emission!(traj::Vector{OfTrajS}, nsteps::Int, Image::Matrix{Float64}, I::Int, J::Int, freq::Float64, bhspin::Float64, model::AbstractModel, data=nothing)
    Xi = MVector{4,Float64}(undef)
    Kconi = MVector{4,Float64}(undef)
    Xf = MVector{4,Float64}(undef)
    Kconf = MVector{4,Float64}(undef)
    Rh::Float64 = 1 + sqrt(1.0 - bhspin * bhspin)
    for k in 1:Constants.NDIM
        Xi[k] = traj[nsteps].X[k]
        Kconi[k] = traj[nsteps].Kcon[k]
    end
    ji, ki = get_jk(Xi, Kconi, freq, bhspin, model, data)

    Intensity = 0.0
    for nstep = nsteps:-1:2
        for k in 1:Constants.NDIM
            Xi[k] = traj[nstep].X[k]
            Xf[k] = traj[nstep-1].X[k]
            Kconi[k] = traj[nstep].Kcon[k]
            Kconf[k] = traj[nstep-1].Kcon[k]
        end

        if !radiating_region(Xf, model, Rh)
            continue
        end

        jf, kf = get_jk(Xf, Kconf, freq, bhspin, model, data)
        Intensity = approximate_solve(Intensity, ji, ki, jf, kf, traj[nstep-1].dl)

        if isnan(Intensity) || isinf(Intensity)
            @error "NaN or Inf encountered in intensity calculation at pixel ($I, $J)"
            println("Intensity = $Intensity")
            DebugFunctions.print_vector("Kconf =", Kconf)
            DebugFunctions.print_vector("Kconi =", Kconi)

            error("NaN or Inf encountered in intensity calculation")
        end
        ji = jf
        ki = kf
    end
    Image[I, J] = Intensity
end

"""
    get_bk_angle(Kcon, Ucov, Bcon, Bcov)

Compute the angle between the photon 4-momentum and the magnetic field
4-vector, in the fluid frame.

# Arguments
- `Kcon`: Contravariant 4-momentum of the photon.
- `Ucov`: Covariant 4-velocity of the fluid.
- `Bcon`: Contravariant magnetic field 4-vector.
- `Bcov`: Covariant magnetic field 4-vector.

# Returns
- The angle between the photon momentum and the magnetic field.
"""
function get_bk_angle(Kcon, Ucov, Bcon, Bcov)
    norm_B = sqrt(abs(Bcov[1] * Bcon[1] + Bcov[2] * Bcon[2] + Bcov[3] * Bcon[3] + Bcov[4] * Bcon[4]))

    if norm_B == 0
        return π / 2
    end
    norm_K = abs(Kcon[1] * Ucov[1] + Kcon[2] * Ucov[2] + Kcon[3] * Ucov[3] + Kcon[4] * Ucov[4])

    μ = (Kcon[1] * Bcov[1] + Kcon[2] * Bcov[2] + Kcon[3] * Bcov[3] + Kcon[4] * Bcov[4]) / (norm_K * norm_B)
    if abs(μ) > 1.0
        μ /= abs(μ)
    end
    return acos(μ)
end

"""
    approximate_solve(Ii, ji, ki, jf, kf, dl)

Evolve the intensity along a geodesic segment using the trapezoidal-rule
approximate solution to the radiative transfer equation.

# Arguments
- `Ii`: Intensity at the start of the segment.
- `ji`: Emissivity at the start of the segment.
- `ki`: Absorption coefficient at the start of the segment.
- `jf`: Emissivity at the end of the segment.
- `kf`: Absorption coefficient at the end of the segment.
- `dl`: Length of the segment along the geodesic.

# Returns
- The intensity at the end of the segment.
"""
function approximate_solve(Ii, ji, ki, jf, kf, dl)
    If = 0.0
    javg = (ji + jf) / 2.0
    kavg = (ki + kf) / 2.0

    dtau = dl * kavg

    if dtau < 1.e-3
        If = Ii + (javg - Ii * kavg) * dl * (1.0 - (dtau / 2.0) * (1.0 - dtau / 3.0))
    else
        efac = exp(-dtau)
        If = Ii * efac + (javg / kavg) * (1.0 - efac)
    end

    return If
end

end
