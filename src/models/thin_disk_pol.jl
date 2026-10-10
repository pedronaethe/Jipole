"""
Polarized emission of the thin-disk model: the model side of
`polarization.jl` for `ThinDiskParams`, translating the polarized parts of
ipole's `model/thin_disk/model.c`.

The disk does not emit along the ray. Its radiation enters as a boundary
condition where the ray crosses the midplane: the Stokes parameters of a
semi-infinite scattering atmosphere (Chandrasekhar 1960, table XXIV), with the
linear polarization lying in the plane of the disk. From there the coherency
tensor is only parallel-transported to the camera.

The temperature profile, Planck function and Chandrasekhar table are the ones
of `thin_disk.jl`, reused as they are. The midplane-crossing test and the
fluid frame are re-expressed here for `SVector`s, because the existing
`thindisk_region`, `get_model_fourv` and `calc_polvec` accept and return
`MVector`s (which allocate) and return the intensity only.
"""
module ThinDiskPol

using StaticArrays
using ..GeoTypes
using ..Coordinates
using ..Metrics
using ..Radiation
using ..Polarization
using ..ThinDisk

export crosses_disk, disk_frame, disk_stokes, disk_stop_index

"""
    crosses_disk(Xi, Xf, model)

Whether the step from `Xi` to `Xf` crosses the midplane inside the emitting
part of the disk, `r_isco < r < Rout` (ipole's `thindisk_region`; the same test
as `ThinDisk.thindisk_region`, for any vector type).

# Arguments
- `Xi`, `Xf`: Start and end of the step, in internal coordinates.
- `model`: Thin disk model parameters.

# Returns
- `true` if the step crosses the emitting region of the disk.
"""
@inline function crosses_disk(Xi, Xf, model::ThinDisk.ThinDiskParams)
    _, th_i = Coordinates.bl_coord(Xi, model)
    r_f, th_f = Coordinates.bl_coord(Xf, model)
    midplane = sign(th_i - π / 2) != sign(th_f - π / 2)
    em_region = r_f > model.r_isco && r_f < model.Rout
    return midplane && em_region
end

"""
    disk_frame(X, bhspin, model)

Rest frame of the disk surface at `X` (ipole's thin-disk `get_model_fourv`):
the four-velocity of the circular orbit with the angular velocity of
`ThinDisk.thindisk_vals`, and the unit normal to the disk (ipole's
`calc_polvec`), which takes the place of the magnetic field in the plasma
tetrad so that tetrad leg 3 is the projected disk normal.

# Arguments
- `X`: Position four-vector in internal coordinates.
- `bhspin`: Dimensionless black hole spin parameter.
- `model`: Thin disk model parameters.

# Returns
- A tuple `(gcov, Ucon, Ucov, Ncon, Ncov)`: the metric at `X`, the
  four-velocity and the disk normal (contravariant and covariant).
"""
@inline function disk_frame(X, bhspin, model::ThinDisk.ThinDiskParams)
    gcov = Metrics.gcov_func(X, bhspin, model)
    r, _ = Coordinates.bl_coord(X, model)
    _, omega = ThinDisk.thindisk_vals(r, bhspin, model)

    T = eltype(gcov)
    u1 = sqrt(-1.0 / (gcov[1, 1] + 2.0 * gcov[1, 4] * omega + gcov[4, 4] * omega * omega))
    Ucon = SVector{4,T}(u1, zero(T), zero(T), omega * u1)
    Ucov = Coordinates.flip_index(Ucon, gcov)

    # Disk normal: the Boyer-Lindquist θ direction, (0, 0, 1, 0). The BL -> KS
    # transformation leaves it unchanged, so only the Jacobian to the internal
    # coordinates acts on it; then it is normalized to unit length.
    dxdX = Coordinates.set_ks_jacobian(X, model)
    f = SVector{4,T}(dxdX[1, 3], dxdX[2, 3], dxdX[3, 3], dxdX[4, 3])
    fcov = Coordinates.flip_index(f, gcov)
    normf = sqrt(f[1] * fcov[1] + f[2] * fcov[2] + f[3] * fcov[3] + f[4] * fcov[4])
    Ncon = f / normf
    Ncov = Coordinates.flip_index(Ncon, gcov)

    return gcov, Ucon, Ucov, Ncon, Ncov
end

"""
    disk_stokes(X, Kcon, Ucov, Ncon, Ncov, bhspin, model)

Invariant Stokes parameters emitted by the disk surface at `X` towards `Kcon`
(ipole's `get_model_stokes` + `fbbpolemis`): a colour-corrected blackbody,
limb-darkened and linearly polarized following Chandrasekhar's table.

`SQ >= 0` along tetrad leg 2, i.e. perpendicular to the projected disk normal:
the polarization lies in the plane of the disk. `SU = SV = 0`.

# Arguments
- `X`: Position four-vector in internal coordinates, at the crossing.
- `Kcon`: Contravariant photon 4-momentum, at the crossing.
- `Ucov`, `Ncon`, `Ncov`: Disk frame, from [`disk_frame`](@ref).
- `bhspin`: Dimensionless black hole spin parameter.
- `model`: Thin disk model parameters.

# Returns
- A tuple `(SI, SQ, SU, SV)`.
"""
@inline function disk_stokes(X, Kcon, Ucov, Ncon, Ncov, bhspin, model::ThinDisk.ThinDiskParams)
    r, _ = Coordinates.bl_coord(X, model)
    Rh = 1 + sqrt(1.0 - bhspin * bhspin)

    T = promote_type(eltype(X), eltype(Kcon))
    z = zero(T)
    r > Rh || return z, z, z, z

    Temp, _ = ThinDisk.thindisk_vals(r, bhspin, model)

    # Cosine of the emission angle to the disk normal, in the disk frame.
    mu = abs(cos(Radiation.get_bk_angle(Kcon, Ucov, Ncon, Ncov)))
    nu = Radiation.get_fluid_nu(Kcon, Ucov)

    # Colour-corrected blackbody times the limb-darkening and polarization fraction.
    SI = model.f^(-4.0) * ThinDisk.bnu(nu, Temp * model.f)
    interpI, interpDel = ThinDisk.interp_chandra(mu)
    SI = SI * interpI
    SQ = SI * interpDel

    # Make the intensities invariant.
    SI = SI / (nu * nu * nu)
    SQ = SQ / (nu * nu * nu)
    return SI, SQ, z, z
end

"""
    Polarization.apply_boundary(N, Xi, Xf, Kconf, bhspin, model::ThinDiskParams)

Thin-disk boundary condition (ipole's `#if THIN_DISK` block in
`integrate_emission`): when the step crosses the emitting part of the midplane,
replace the Stokes parameters carried by the ray with those emitted by the disk
surface, in the disk frame at `Xf`.

As in ipole, only the Stokes block of the tetrad-frame tensor is replaced. The
disk is made opaque by where the integration starts, not here: see
[`disk_stop_index`](@ref).
"""
@inline function Polarization.apply_boundary(N, Xi, Xf, Kconf, bhspin, model::ThinDisk.ThinDiskParams)
    crosses_disk(Xi, Xf, model) || return N

    gcov, Ucon, Ucov, Ncon, Ncov = disk_frame(Xf, bhspin, model)
    _, Econ, Ecov = Polarization.plasma_tetrad(Ucon, Kconf, Ncon, gcov)
    SI, SQ, SU, SV = disk_stokes(Xf, Kconf, Ucov, Ncon, Ncov, bhspin, model)

    N_tetrad = Polarization.complex_coord_to_tetrad_rank2(N, Ecov)
    N_tetrad = Polarization.stokes_to_tensor(N_tetrad, SI, SQ, SU, SV)
    return Polarization.complex_tetrad_to_coord_rank2(N_tetrad, Econ)
end

"""
    disk_stop_index(traj, nsteps, model)

Number of trajectory points ipole would have stored for this ray. ipole treats
the disk as opaque by ending the backward geodesic integration shortly behind
it (the `THIN_DISK` block of `stop_backward_integration`): once a step ends
across the emitting midplane from its half-step point, three more points are
taken and the integration stops. Jipole's `trace_geodesic` has no such rule and
runs on to the horizon or the outer boundary, so the polarized integration
starts from this index instead of from the end of the stored trajectory. The
geodesic itself is identical up to that point.

Note that the rule only sees crossings in the second half of a step. A ray that
crosses in the first half of a step is not stopped there, in ipole or here.

# Arguments
- `traj`: Geodesic trajectory (`traj[1]` is the camera).
- `nsteps`: Number of points in the trajectory.
- `model`: Thin disk model parameters.

# Returns
- The index of the last trajectory point to use, `<= nsteps`.
"""
function disk_stop_index(traj::Vector{GeoTypes.OfTrajGeneric{T}}, nsteps::Int, model::ThinDisk.ThinDiskParams) where {T}
    n_left = -1
    @inbounds for k in 1:nsteps
        if n_left < 0 && crosses_disk(traj[k].X, traj[k].Xhalf, model)
            n_left = 2          # set the timer when we reach the disk
        elseif n_left > 0
            n_left -= 1         # or decrement it if it is set
        elseif n_left == 0
            return k            # timer at zero: ipole stops here
        end
    end
    return nsteps
end

"""
    Polarization.get_pol_state(X, Kcon, freq, bhspin, model::ThinDiskParams, data)

The thin disk has no emission, absorption or rotation along the ray
(`Radiation.radiating_region` is always `false` for it), so this method is
never reached. It exists so that the transfer loop is fully typed for this
model, and returns vanishing coefficients.
"""
@inline function Polarization.get_pol_state(X, Kcon, freq, bhspin, model::ThinDisk.ThinDiskParams, data)
    T = promote_type(eltype(X), eltype(Kcon))
    z = zero(SVector{4,T})
    return zero(Polarization.PolCoeffs{T}), SVector{4,T}(one(T), zero(T), zero(T), zero(T)), z, z
end

"""
    Radiation.integrate_emission!(traj, nsteps, Image, I, J, freq, bhspin, model::ThinDiskParams, sink::PolSink)

Polarized pixel integration for the thin disk: the unpolarized intensity from
the existing thin-disk method, then the polarized pixel from
`Polarization.integrate_emission_pol`, started behind the disk at
[`disk_stop_index`](@ref).

(The thin disk has its own Stokes-I method of `Radiation.integrate_emission!`,
so the generic `PolSink` method would be ambiguous with it; this method also
resolves that.)
"""
function Radiation.integrate_emission!(traj::Vector{GeoTypes.OfTrajGeneric{T}}, nsteps::Int, Image, I, J, freq, bhspin, model::ThinDisk.ThinDiskParams, sink::Polarization.PolSink) where {T}
    Radiation.integrate_emission!(traj, nsteps, Image, I, J, freq, bhspin, model, sink.data)
    nsteps_pol = disk_stop_index(traj, nsteps, model)
    SI, SQ, SU, SV, tauF = Polarization.integrate_emission_pol(traj, nsteps_pol, freq, bhspin, model, sink.data)
    Polarization.save_pixel!(sink.pol, I, J, SI, SQ, SU, SV, tauF, freq, sink.qu_conv)
    return nothing
end

end
