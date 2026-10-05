"""
Polarized transfer for GRMHD snapshots (`Iharm`, and `Kharma`, which shares
`IharmParams`/`IharmData`): the model side of `polarization.jl`.

It adds the method of `Polarization.get_pol_state` for `IharmParams`, composed
from the interpolation helpers that the Stokes-I path already uses, and the
automatic-differentiation entry point for polarized images.
"""
module IharmPol

using StaticArrays
using ForwardDiff
using ..Constants
using ..Grid
using ..Radiation
using ..Polarization
using ..Imaging
using ..ImagingPol
using ..Iharm

export plasma_state, dual_problem, calculate_gradients_pol

"""
    plasma_state(X, Kcon, bhspin, model, data)

Local plasma quantities that set the synchrotron transfer coefficients at `X`:
the zone lookup, the fluid four-velocity and field, and from them the electron
density, fluid-frame frequency, field strength, electron temperature and
field-wavevector angle. Every value comes from the existing `Iharm` helpers,
called in the order ipole's `jar_calc_dist` uses.

(`Iharm.jar_calc` evaluates the same quantities for Stokes I before applying
the Leung et al. fit; it could be rewritten on top of this function.)

# Arguments
- `X`: Position four-vector in internal coordinates.
- `Kcon`: Contravariant photon 4-momentum.
- `bhspin`: Dimensionless black hole spin parameter.
- `model`: Iharm model parameters.
- `data`: GRMHD snapshot(s).

# Returns
- A named tuple `(Ne, nu, θe, B, θ, Ucon, Ucov, Bcon, Bcov)`. Where `Ne <= 0`
  (outside the grid or above the magnetization cut) the other scalars are
  not evaluated and are returned as zero.
"""
@inline function plasma_state(X, Kcon, bhspin, model::Iharm.IharmParams, data)
    zone = Grid.locate(X, model)
    Ne = Iharm.get_model_ne(zone, X, model, data)
    Ucon, Ucov, Bcon, Bcov = Iharm.get_model_fourv(zone, X, Kcon, bhspin, model, data)

    # Common number type of everything returned, so both branches agree.
    T = promote_type(typeof(Ne), eltype(Ucov), eltype(Bcov), eltype(Kcon),
        eltype(data[1].b), eltype(data[1].θe), typeof(zone.del2))
    z = zero(T)

    if !(Ne > 0)
        return (Ne=z, nu=z, θe=z, B=z, θ=z, Ucon=Ucon, Ucov=Ucov, Bcon=Bcon, Bcov=Bcov)
    end

    nu = Radiation.get_fluid_nu(Kcon, Ucov)
    θ = Radiation.get_bk_angle(Kcon, Ucov, Bcon, Bcov)
    B = Iharm.get_model_b(zone, X, model, data)
    θe = Iharm.get_model_thetae(zone, X, model, data)

    return (Ne=T(Ne), nu=T(nu), θe=T(θe), B=T(B), θ=T(θ), Ucon=Ucon, Ucov=Ucov, Bcon=Bcon, Bcov=Bcov)
end

"""
    Polarization.get_pol_state(X, Kcon, freq, bhspin, model::IharmParams, data)

Polarized transfer coefficients and plasma frame of the GRMHD model at `X`:
thermal synchrotron coefficients (`Polarization.thermal_jar`) for the local
[`plasma_state`](@ref). `freq` is unused (the fluid-frame frequency follows
from `Kcon`) and is kept for symmetry with `Radiation.get_jk`.
"""
@inline function Polarization.get_pol_state(X, Kcon, freq, bhspin, model::Iharm.IharmParams, data)
    s = plasma_state(X, Kcon, bhspin, model, data)
    coeffs = Polarization.thermal_jar(s.Ne, s.nu, s.θe, s.B, s.θ)
    return coeffs, s.Ucon, s.Bcon, s.Bcov
end

"""
    dual_problem(ctx; MBH, Rhigh, Rlow, beta_crit, th_beg, sigma_cut, sigma_cut_high,
                 M_unit, ro, th, phi, sourceD, wrt=())

Build the dual-number copy of a GRMHD problem for forward-mode automatic
differentiation: every parameter named in `wrt` is seeded with its own partial,
the units and the model are rebuilt from the seeded values, and the derived
plasma arrays (`ne`, `θe`, `b`, `sigma`, `beta`) are recomputed so that they
carry the partials too.

This is the setup stage of `Iharm.calculate_gradients`, isolated so that the
polarized gradient ([`calculate_gradients_pol`](@ref)) can share it; the
construction is identical. `Iharm.calculate_gradients` could call this function
instead of repeating it.

# Arguments
- `ctx`: A context from `Iharm.grmhd_context`.
- keyword arguments: values of the differentiable parameters (see
  `Iharm.GRADIENT_PARAM_NAMES`); `wrt` lists the ones to differentiate.

# Returns
- A named tuple `(model, data, ro, th, phi, sourceD)` of dual-typed objects,
  with `data` a one-element vector as `Imaging.raytrace_image` expects.
"""
function dual_problem(ctx; MBH, Rhigh, Rlow, beta_crit, th_beg, sigma_cut, sigma_cut_high, M_unit,
        ro, th, phi, sourceD, wrt::NTuple{N,Symbol}=()) where N

    dualize(val, sym) = sym in wrt ? ForwardDiff.Dual{Nothing,Float64,N}(val, ForwardDiff.Partials(ntuple(i -> i == findfirst(==(sym), wrt) ? 1.0 : 0.0, N))) : val

    MBH_d, Rhigh_d, Rlow_d = dualize(MBH, :MBH), dualize(Rhigh, :Rhigh), dualize(Rlow, :Rlow)
    beta_crit_d, th_beg_d = dualize(beta_crit, :beta_crit), dualize(th_beg, :th_beg)
    sigma_cut_d, sigma_cut_high_d = dualize(sigma_cut, :sigma_cut), dualize(sigma_cut_high, :sigma_cut_high)
    M_unit_d = dualize(M_unit, :M_unit)
    ro_d, th_d, phi_d, sourceD_d = dualize(ro, :ro), dualize(th, :th), dualize(phi, :phi), dualize(sourceD, :sourceD)

    T = promote_type(typeof(MBH_d), typeof(Rhigh_d), typeof(Rlow_d), typeof(beta_crit_d), typeof(th_beg_d),
                      typeof(sigma_cut_d), typeof(sigma_cut_high_d), typeof(M_unit_d),
                      typeof(ro_d), typeof(th_d), typeof(phi_d), typeof(sourceD_d))

    L_unit_d = Constants.GNEWT * MBH_d * Constants.MSUN / Constants.CL^2
    T_unit_d = L_unit_d / Constants.CL
    RHO_unit_d = M_unit_d / L_unit_d^3
    U_unit_d = RHO_unit_d * Constants.CL^2
    B_unit_d = Constants.CL * sqrt(4π * RHO_unit_d)

    model_d = Iharm.IharmParams{T}(ctx.metric, ctx.ELECTRONS, ctx.RADIATION,
        ctx.gam, ctx.game, ctx.gamp, (0.0), (0.0),
        ctx.mu_i, ctx.mu_e, ctx.mu_tot, ctx.Ne_factor,
        M_unit_d, T_unit_d, L_unit_d, MBH_d, ctx.tp_over_te,
        RHO_unit_d, U_unit_d, B_unit_d, ctx.a, ctx.hslope, ctx.Rin, ctx.Rout,
        ctx.poly_xt, ctx.poly_alpha, ctx.mks_smooth, ctx.poly_norm,
        ctx.mks3R0, ctx.mks3H0, ctx.mks3MY1, ctx.mks3MY2, ctx.mks3MP0,
        ctx.N1, ctx.N2, ctx.N3, ctx.dx, ctx.startx, ctx.stopx, ctx.cstartx, ctx.cstopx,
        ctx.rmin_geo, ctx.rmax_geo, th_beg_d, Rlow_d, Rhigh_d, beta_crit_d, sigma_cut_d,
        sigma_cut_high_d, ctx.slow_light)

    b_d = ctx.b_normalized .* T(B_unit_d)
    data_d = Iharm.IharmData(ctx.t, ctx.RHO, ctx.UU, ctx.U1, ctx.U2, ctx.U3, ctx.B1, ctx.B2, ctx.B3, similar(ctx.RHO, T), b_d, similar(ctx.RHO, T), similar(ctx.RHO, T), similar(ctx.RHO, T), similar(ctx.RHO, T))
    Iharm.init_physical_quantities([data_d], 1, model_d, Rhigh_d)

    return (model=model_d, data=[data_d], ro=ro_d, th=th_d, phi=phi_d, sourceD=sourceD_d)
end

"""
    calculate_gradients_pol(ctx, freq, pixels_x, pixels_y, fovx_uas, maxnstep, xoff, yoff;
        MBH, Rhigh, Rlow, beta_crit, th_beg, sigma_cut, sigma_cut_high, M_unit,
        ro, th, phi, sourceD, wrt=())

Polarized counterpart of `Iharm.calculate_gradients`: the unpolarized image,
the polarized image, and their derivatives with respect to the parameters in
`wrt`, from one forward-mode pass of dual numbers through the ordinary
ray-tracing code (`ImagingPol.raytrace_image_pol`).

# Arguments
- as `Iharm.calculate_gradients`.

# Returns
- `(Image, pol)` if `wrt` is empty, otherwise `(Image, pol, grads, grads_pol)`:
  `Image` is `(nx, ny)`, `pol` is `(NIMG, nx, ny)` (I, Q, U, V, Faraday depth),
  and `grads`/`grads_pol` are named tuples with one array of the same shape per
  parameter, holding `∂Image/∂p` and `∂pol/∂p`.
"""
function calculate_gradients_pol(ctx, freq, pixels_x, pixels_y, fovx_uas, maxnstep, xoff, yoff;
        MBH, Rhigh, Rlow, beta_crit, th_beg, sigma_cut, sigma_cut_high, M_unit,
        ro, th, phi, sourceD, wrt::NTuple{N,Symbol}=()) where N

    p = dual_problem(ctx; MBH=MBH, Rhigh=Rhigh, Rlow=Rlow, beta_crit=beta_crit, th_beg=th_beg,
        sigma_cut=sigma_cut, sigma_cut_high=sigma_cut_high, M_unit=M_unit,
        ro=ro, th=th, phi=phi, sourceD=sourceD, wrt=wrt)

    Dxsize = p.sourceD / p.model.L_unit / Constants.MUAS_PER_RAD * fovx_uas
    fovx_d = Dxsize / p.ro
    fovy_d = Dxsize / p.ro
    Rh = 1.0 + sqrt(1.0 - ctx.a^2)

    Image_dual, pol_dual = ImagingPol.raytrace_image_pol(p.model, p.data, p.ro, p.th, p.phi, freq, pixels_x, pixels_y,
                                 fovx_d, fovy_d, maxnstep, Rh, xoff, yoff)

    Image = ForwardDiff.value.(Image_dual)
    pol = ForwardDiff.value.(pol_dual)
    N == 0 && return Image, pol
    grads = NamedTuple{wrt}(Tuple(ForwardDiff.partials.(Image_dual, i) for i in 1:N))
    grads_pol = NamedTuple{wrt}(Tuple(ForwardDiff.partials.(pol_dual, i) for i in 1:N))
    return Image, pol, grads, grads_pol
end

end
