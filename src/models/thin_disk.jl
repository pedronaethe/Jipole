"""
 Thin disk (Novikov-Thorne / Krolik & Hawley) model. The disk does not emit along the
ray: its radiation enters where the ray crosses the midplane, as the Stokes parameters
of a semi-infinite scattering atmosphere (Chandrasekhar 1960, table XXIV), with the
linear polarization in the plane of the disk.
"""
module ThinDisk

using StaticArrays
using ..Constants
using ..GeoTypes
using ..AbstractModels
using ..Coordinates
using ..Metrics
using ..Radiation
using ..Polarization
using ..Tetrads

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
    crosses_disk(Xi, Xf, model)

Whether the step from `Xi` to `Xf` crosses the midplane inside the emitting
part of the disk, `r_isco < r < Rout` (ipole's `thindisk_region`).

# Arguments
- `Xi`, `Xf`: Start and end of the step, in internal coordinates.
- `model`: Thin disk model parameters.

# Returns
- `true` if the step crosses the emitting region of the disk.
"""
@inline function crosses_disk(Xi, Xf, model::ThinDiskParams)
    _, th_i = Coordinates.bl_coord(Xi, model)
    r_f, th_f = Coordinates.bl_coord(Xf, model)
    midplane = sign(th_i - π / 2) != sign(th_f - π / 2)
    em_region = r_f > model.r_isco && r_f < model.Rout
    return midplane && em_region
end

"""
    Radiation.radiating_region(X, model::ThinDiskParams, Rh)

Always `false`: the thin disk has no emission along the ray, only where the ray
crosses the disk (see [`crosses_disk`](@ref) and [`disk_surface`](@ref)).
"""
function Radiation.radiating_region(X, model::ThinDiskParams, Rh::Float64)
    return false
end

"""
      thindisk_vals(r, bhspin, model)

  Local disk temperature, with the Krolik & Hawley (2002) relativistic correction,
  and orbital angular frequency at radius `r`.

  # Arguments
  - `r`: Radial coordinate, in Boyer-Lindquist/Kerr-Schild coordinates.
  - `bhspin`: Dimensionless black hole spin parameter.
  - `model`: Thin disk model parameters.

  # Returns
  - A tuple `(Temp, omega)`.
  """
function thindisk_vals(r::Float64, bhspin::Float64, model::ThinDiskParams)
    # Krolik & Hawley (2002) correction factor
    y = sqrt(r)
    yms = sqrt(model.r_isco)
    y1 = 2.0 * cos(1.0 / 3.0 * (acos(bhspin) - π))
    y2 = 2.0 * cos(1.0 / 3.0 * (acos(bhspin) + π))
    y3 = -2.0 * cos(1.0 / 3.0 * acos(bhspin))
    arg1 = 3.0 * bhspin / (2.0 * y)
    arg2 = 3.0 * (y1 - bhspin)^2 / (y * y1 * (y1 - y2) * (y1 - y3))
    arg3 = 3.0 * (y2 - bhspin)^2 / (y * y2 * (y2 - y1) * (y2 - y3))
    arg4 = 3.0 * (y3 - bhspin)^2 / (y * y3 * (y3 - y1) * (y3 - y2))
    kc = 1.0 - yms / y - arg1 * log(y / yms) - arg2 * log((y - y1) / (yms - y1)) - arg3 * log((y - y2) / (yms - y2)) - arg4 * log((y - y3) / (yms - y3))

    b = 1.0 - 3.0 / r + 2.0 * bhspin / r^(3 / 2)
    d = r * r - 2.0 * r + bhspin * bhspin
    lc = (model.r_isco * model.r_isco - 2.0 * bhspin * sqrt(model.r_isco) + bhspin * bhspin) / ((model.r_isco)^(1.5) - 2.0 * sqrt(model.r_isco) + bhspin)
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
    disk_surface(X, Kcon, bhspin, model)

What the disk surface at `X` emits towards `Kcon` (ipole's thin-disk
`get_model_fourv`, `calc_polvec`, `get_model_stokes` and `fbbpolemis`): a
colour-corrected blackbody, limb-darkened and linearly polarized following
Chandrasekhar's table (1960, table XXIV), in the rest frame of the circular
orbit at `X`.

The disk normal takes the place of the magnetic field in the plasma tetrad, so
tetrad leg 3 is the projected normal and `SQ >= 0` along leg 2: the polarization
lies in the plane of the disk (`SU = SV = 0`).

# Arguments
- `X`: Position four-vector in internal coordinates, at the crossing.
- `Kcon`: Contravariant photon 4-momentum, at the crossing.
- `bhspin`: Dimensionless black hole spin parameter.
- `model`: Thin disk model parameters.

# Returns
- `(SI, SQ, gcov, Ucon, Ncon)`: invariant Stokes I and Q, and the frame for the
tetrad (metric, four-velocity, disk normal). Inside the horizon the Stokes
parameters and the frame vectors are zero.
"""
@inline function disk_surface(X, Kcon, bhspin, model::ThinDiskParams)
    r, _ = Coordinates.bl_coord(X, model)
    Rh = 1 + sqrt(1.0 - bhspin * bhspin)
    gcov = Metrics.gcov_func(X, bhspin, model)
    T = eltype(gcov)
    Ts = promote_type(eltype(X), eltype(Kcon))
    r > Rh || return zero(Ts), zero(Ts), gcov, zero(SVector{4,T}), zero(SVector{4,T})

    Temp, omega = thindisk_vals(r, bhspin, model)

    # Four-velocity of the circular orbit.
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

    # Cosine of the emission angle to the disk normal, and frequency, in the disk frame.
    mu = abs(cos(Radiation.get_bk_angle(Kcon, Ucov, Ncon, Ncov)))
    nu = Radiation.get_fluid_nu(Kcon, Ucov)

    # Colour-corrected blackbody: f^-4 times the Planck function at temperature f*Temp.
    Tc = Temp * model.f
    SI = model.f^(-4.0) * (2 * Constants.HPL * nu^3 / (Constants.CL^2) / (exp(Constants.HPL * nu / (Constants.KBOL * Tc)) - 1))

    # Limb darkening and polarization fraction: linear interpolation in Chandrasekhar's table.
    j = 1
    while CH_TABLE.mu[j] < mu
        j += 1
    end
    j -= 1
    w = (mu - CH_TABLE.mu[j]) / (CH_TABLE.mu[j+1] - CH_TABLE.mu[j])
    interpI = (1.0 - w) * CH_TABLE.I[j] + w * CH_TABLE.I[j+1]
    interpDel = (1.0 - w) * CH_TABLE.delta[j] + w * CH_TABLE.delta[j+1]

    SI = SI * interpI
    SQ = SI * interpDel

    # Make the intensities invariant.
    SI = SI / (nu * nu * nu)
    SQ = SQ / (nu * nu * nu)
    return SI, SQ, gcov, Ucon, Ncon
end

"""
    Radiation.integrate_emission!(traj, nsteps, Image, I, J, freq, bhspin, model::ThinDiskParams, data=nothing)

Stokes I pixel of the thin disk. There is no emission along the ray: `Image[I, J]`
is the surface intensity ([`disk_surface`](@ref)) at the disk crossing closest to
the camera. `data` is unused by this model.
"""
function Radiation.integrate_emission!(traj::Vector{GeoTypes.OfTrajGeneric{T}}, nsteps::Int, Image, I, J, freq, bhspin, model::ThinDiskParams, data=nothing) where {T}
    Intensity = zero(T)
    for nstep = nsteps:-1:2
        Xi = traj[nstep].X
        Kconi = traj[nstep].Kcon
        if crosses_disk(Xi, traj[nstep-1].X, model)
            Intensity, _ = disk_surface(Xi, Kconi, bhspin, model)
        end
    end
    Image[I, J] = Intensity
end

"""
    Polarization.apply_boundary(N, Xi, Xf, Kconf, bhspin, model::ThinDiskParams)

Thin-disk boundary condition (ipole's `#if THIN_DISK` block in
`integrate_emission`): when the step crosses the emitting part of the midplane,
replace the Stokes parameters carried by the ray with those emitted by the disk
surface, in the disk frame at `Xf`. As in ipole, only the Stokes block of the
tetrad-frame tensor is replaced.
"""
@inline function Polarization.apply_boundary(N, Xi, Xf, Kconf, bhspin, model::ThinDiskParams)
    crosses_disk(Xi, Xf, model) || return N

    SI, SQ, gcov, Ucon, Ncon = disk_surface(Xf, Kconf, bhspin, model)
    _, Econ, Ecov = Tetrads.make_plasma_tetrad(Ucon, Kconf, Ncon, gcov)

    N_tetrad = Polarization.complex_coord_to_tetrad_rank2(N, Ecov)
    N_tetrad = Polarization.stokes_to_tensor(N_tetrad, SI, SQ, zero(SI), zero(SI))
    return Polarization.complex_tetrad_to_coord_rank2(N_tetrad, Econ)
end

"""
    Radiation.integrate_emission!(traj, nsteps, Image, I, J, freq, bhspin, model::ThinDiskParams, sink::PolSink)

Polarized pixel of the thin disk: the Stokes I pixel from the method above, then
the polarized pixel from `Polarization.integrate_emission_pol`, started behind
the disk where ipole would have stopped the ray.

ipole makes the disk opaque by ending the backward integration three points after
a step whose half-step point lies across the emitting midplane (the `THIN_DISK`
block of its `stop_backward_integration`). Jipole's tracer runs on, so the
polarized integration starts from that point instead. Crossings in the first half
of a step are not seen, in ipole or here.

(The generic `PolSink` method would be ambiguous with the thin disk's own
Stokes I method; this method also resolves that.)
"""
function Radiation.integrate_emission!(traj::Vector{GeoTypes.OfTrajGeneric{T}}, nsteps::Int, Image, I, J, freq, bhspin, model::ThinDiskParams, sink::Polarization.PolSink) where {T}
    Radiation.integrate_emission!(traj, nsteps, Image, I, J, freq, bhspin, model, sink.data)

    nsteps_pol = nsteps
    n_left = -1
    @inbounds for k in 1:nsteps
        if n_left < 0 && crosses_disk(traj[k].X, traj[k].Xhalf, model)
            n_left = 2          # set the timer when we reach the disk
        elseif n_left > 0
            n_left -= 1         # or decrement it if it is set
        elseif n_left == 0
            nsteps_pol = k      # timer at zero: ipole stops here
            break
        end
    end

    SI, SQ, SU, SV, tauF = Polarization.integrate_emission_pol(traj, nsteps_pol, freq, bhspin, model, sink.data)
    Polarization.save_pixel!(sink.pol, I, J, SI, SQ, SU, SV, tauF, freq, sink.qu_conv)
    return nothing
end


end
