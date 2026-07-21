"""
Photon geodesic integration in Kerr-Schild-family coordinates.

The connection-coefficient dispatch (`get_connection_analytic`) branches
on `model.metric` directly rather than on the model's type: every model's
parameters carry a `metric` (metric family) and `hslope` (polar
compression) field, and the plain-MKS connection formula
([`Models_and_MKS_connection_analytic`](@ref)) is mathematically exact
for `hslope == 1.0`, which is what `Analytic`/`ThinDisk` always use. Only
`Iharm` ever uses `hslope != 1.0` or the FMKS branch.
"""
module Geodesics

using StaticArrays
using LinearAlgebra
using ..Constants
using ..GeoTypes
using ..Coordinates
using ..Metrics
using ..Tetrads
using ..DebugFunctions

export init_XK!, init_Kcon, get_pixel, CalculateGeodesics, Models_and_MKS_connection_analytic,
    FMKS_connection_analytic, get_connection_analytic, get_connection_analytic!,
    compute_dKcon, push_photon, push_photon!, get_connection, stepsize,
    stop_backward_integration, trace_geodesic

"""
    init_XK!(X, Kcon, i, j, Xcam, nx, ny, fovx, fovy, bhspin, model, xoff=0, yoff=0)

Initialize the position and photon 4-momentum for the geodesic launched
from camera pixel `(i, j)`.

# Arguments
- `X`: Output vector, overwritten with the initial position.
- `Kcon`: Output vector, overwritten with the initial 4-momentum.
- `i`, `j`: Pixel indices in the image plane.
- `Xcam`: Camera position in internal coordinates.
- `nx`, `ny`: Image resolution.
- `fovx`, `fovy`: Field of view, in radians.
- `bhspin`: Dimensionless black hole spin parameter.
- `model`: Model parameters.
- `xoff`, `yoff`: Image plane offsets.
"""
function init_XK!(X::AbstractVector{T}, Kcon::AbstractVector{T}, i::Int, j::Int, Xcam::AbstractVector{T}, nx::Int, ny::Int, fovx, fovy, bhspin, model, xoff=0, yoff=0) where {T}
    _, Econ, Ecov = Tetrads.make_camera_tetrad(Xcam, bhspin, model)
    dxoff::Float64 = (i + 0.5 + xoff - 0.01) / nx - 0.5
    dyoff::Float64 = (j + 0.5 + yoff) / ny - 0.5

    Kcon_tetrad = SVector{4,T}(
        zero(T),
        (dxoff * cos(zero(T)) - dyoff * sin(zero(T))) * fovx,
        (dxoff * sin(zero(T)) + dyoff * cos(zero(T))) * fovy,
        one(T)
    )

    Kcon_tetrad = Tetrads.null_normalize(Kcon_tetrad, one(T))

    Tetrads.tetrad_to_coordinate!(Kcon, Econ, Kcon_tetrad)

    @inbounds for mu in 1:Constants.NDIM
        X[mu] = Xcam[mu]
    end
end

"""
    init_Kcon(i, j, Xcam, nx, ny, fovx, fovy, bhspin, model, xoff=0, yoff=0)

Non-mutating, GPU-safe variant of [`init_XK!`](@ref) that returns the
initial photon 4-momentum directly. The initial position is always
`Xcam`, so no position output is needed.

# Arguments
- `i`, `j`: Pixel indices in the image plane.
- `Xcam`: Camera position in internal coordinates.
- `nx`, `ny`: Image resolution.
- `fovx`, `fovy`: Field of view, in radians.
- `bhspin`: Dimensionless black hole spin parameter.
- `model`: Model parameters.
- `xoff`, `yoff`: Image plane offsets.

# Returns
- The initial photon 4-momentum.
"""
function init_Kcon(i::Int, j::Int, Xcam::AbstractVector{T}, nx::Int, ny::Int, fovx, fovy, bhspin, model, xoff=0, yoff=0) where {T}
    _, Econ, Ecov = Tetrads.make_camera_tetrad(Xcam, bhspin, model)
    dxoff::Float64 = (i + 0.5 + xoff - 0.01) / nx - 0.5
    dyoff::Float64 = (j + 0.5 + yoff) / ny - 0.5

    Kcon_tetrad = SVector{4,T}(
        zero(T),
        (dxoff * cos(zero(T)) - dyoff * sin(zero(T))) * fovx,
        (dxoff * sin(zero(T)) + dyoff * cos(zero(T))) * fovy,
        one(T)
    )

    Kcon_tetrad = Tetrads.null_normalize(Kcon_tetrad, one(T))

    return Tetrads.tetrad_to_coordinate(Econ, Kcon_tetrad)
end

"""
    get_pixel(traj, i, j, Xcam, fovx, fovy, freq, nx, ny, bhspin, Rh, Rstop, model, xoff=0, yoff=0)

Trace the geodesic for camera pixel `(i, j)`, filling `traj` with the
trajectory steps.

# Arguments
- `traj`: Output trajectory vector, filled by [`trace_geodesic`](@ref).
- `i`, `j`: Pixel indices in the image plane.
- `Xcam`: Camera position in internal coordinates.
- `fovx`, `fovy`: Field of view, in radians.
- `freq`: Unitless photon frequency (already scaled by `Kcon`).
- `nx`, `ny`: Image resolution.
- `bhspin`: Dimensionless black hole spin parameter.
- `Rh`: Event horizon radius.
- `Rstop`: Backward-integration stopping radius.
- `model`: Model parameters.
- `xoff`, `yoff`: Image plane offsets.

# Returns
- A tuple `(nstep, midplane_crossings)`.
"""
function get_pixel(traj::Vector{OfTrajS}, i::Int, j::Int, Xcam::MVector{4,Float64}, fovx::Float64, fovy::Float64, freq::Float64, nx::Int64, ny::Int64, bhspin::Float64, Rh::Float64, Rstop::Float64, model, xoff=0, yoff=0)
    X_mut = MVector{4,Float64}(undef)
    Kcon_mut = MVector{4,Float64}(undef)

    init_XK!(X_mut, Kcon_mut, i, j, Xcam, nx, ny, fovx, fovy, bhspin, model, xoff, yoff)

    X = SVector{4,Float64}(X_mut)
    Kcon = SVector{4,Float64}(Kcon_mut) * freq

    nstep, midplane_crossings = trace_geodesic(X, Kcon, traj, i, j, bhspin, Rh, Rstop, model)

    if nstep >= (50000 - 1)
        @error "Photon orbit trapped at pixel ($i, $j). Reached max $nstep steps."
    end

    return nstep, midplane_crossings
end

"""
    CalculateGeodesics(Xcam, fovx, fovy, freq_cgs, maxnstep, nx, ny, bhspin, Rstop, model)

Trace the geodesics for every pixel of the image plane.

# Arguments
- `Xcam`: Camera position in internal coordinates.
- `fovx`, `fovy`: Field of view, in radians.
- `freq_cgs`: Frequency, in cgs units.
- `maxnstep`: Maximum number of integration steps for each geodesic.
- `nx`, `ny`: Image resolution.
- `bhspin`: Dimensionless black hole spin parameter.
- `Rstop`: Backward-integration stopping radius.
- `model`: Model parameters.

# Returns
- A matrix of geodesic trajectories, one per pixel.
"""
function CalculateGeodesics(Xcam, fovx, fovy, freq_cgs, maxnstep, nx, ny, bhspin, Rstop, model)
    Rh = 1 + sqrt(1.0 - bhspin * bhspin)
    trajs = Matrix{Vector{OfTrajS}}(undef, nx, ny)
    freq_unitless = freq_cgs * Constants.HPL / (Constants.ME * Constants.CL * Constants.CL)

    Threads.@threads for i in 0:(nx-1)
        for j in 0:(ny-1)
            trajs[i+1, j+1] = Vector{OfTrajS}()
            sizehint!(trajs[i+1, j+1], maxnstep)

            nstep, _ = get_pixel(trajs[i+1, j+1], i, j, Xcam, fovx, fovy, freq_unitless, nx, ny, bhspin, Rh, Rstop, model)

            resize!(trajs[i+1, j+1], nstep)
        end
    end
    return trajs
end

"""
    Models_and_MKS_connection_analytic(X, bhspin, model)

Compute the analytic connection coefficients for the plain
Modified-Kerr-Schild metric (`model.hslope`-compressed polar coordinate).
Exact for any `model.hslope`; `Analytic`/`ThinDisk` always use
`hslope == 1.0`, which reduces this to the uncompressed `th = π X[3]`
mapping.

# Arguments
- `X`: Position four-vector in internal coordinates.
- `bhspin`: Dimensionless black hole spin parameter.
- `model`: Model parameters (`model.hslope`).

# Returns
- The connection coefficients `Γ^μ_{αβ}`, as an `SArray`.
"""
@inline function Models_and_MKS_connection_analytic(X::AbstractVector{T}, bhspin, model) where {T}
    lconn = MArray{Tuple{4,4,4},T,3,64}(undef)

    r1 = exp(X[2])
    r2 = r1 * r1
    r3 = r2 * r1
    r4 = r3 * r1

    th = π * X[3] + ((1.0 - model.hslope) / 2.0) * sin(2.0 * π * X[3])
    dthdx2 = π + (1.0 - model.hslope) * π * cos(2.0 * π * X[3])
    d2thdx22 = -2.0 * π^2 * (1.0 - model.hslope) * sin(2.0 * π * X[3])
    dthdx22 = dthdx2 * dthdx2

    sth = sin(th)
    cth = cos(th)
    sth2 = sth * sth
    r1sth2 = r1 * sth2
    sth4 = sth2 * sth2
    cth2 = cth * cth
    cth4 = cth2 * cth2
    s2th = 2.0 * sth * cth
    c2th = 2.0 * cth2 - 1.0

    a2 = bhspin * bhspin
    a2sth2 = a2 * sth2
    a2cth2 = a2 * cth2
    a3 = a2 * bhspin
    a4 = a3 * bhspin
    a4cth4 = a4 * cth4

    rho2 = r2 + a2cth2
    rho22 = rho2 * rho2
    rho23 = rho22 * rho2
    irho2 = 1.0 / rho2
    irho22 = irho2 * irho2
    irho23 = irho22 * irho2
    irho23_dthdx2 = irho23 / dthdx2

    fac1 = r2 - a2cth2
    fac1_rho23 = fac1 * irho23
    fac2 = a2 + 2 * r2 + a2 * c2th
    fac3 = a2 + r1 * (-2.0 + r1)

    lconn[1, 1, 1] = 2.0 * r1 * fac1_rho23
    lconn[1, 1, 2] = r1 * (2.0 * r1 + rho2) * fac1_rho23
    lconn[1, 1, 3] = -a2 * r1 * s2th * dthdx2 * irho22
    lconn[1, 1, 4] = -2.0 * bhspin * r1sth2 * fac1_rho23

    lconn[1, 2, 1] = lconn[1, 1, 2]
    lconn[1, 2, 2] = 2.0 * r2 * (r4 + r1 * fac1 - a4cth4) * irho23
    lconn[1, 2, 3] = -a2 * r2 * s2th * dthdx2 * irho22
    lconn[1, 2, 4] = bhspin * r1 * (-r1 * (r3 + 2 * fac1) + a4cth4) * sth2 * irho23

    lconn[1, 3, 1] = lconn[1, 1, 3]
    lconn[1, 3, 2] = lconn[1, 2, 3]
    lconn[1, 3, 3] = -2.0 * r2 * dthdx22 * irho2
    lconn[1, 3, 4] = a3 * r1sth2 * s2th * dthdx2 * irho22

    lconn[1, 4, 1] = lconn[1, 1, 4]
    lconn[1, 4, 2] = lconn[1, 2, 4]
    lconn[1, 4, 3] = lconn[1, 3, 4]
    lconn[1, 4, 4] = 2.0 * r1sth2 * (-r1 * rho22 + a2sth2 * fac1) * irho23

    lconn[2, 1, 1] = fac3 * fac1 / (r1 * rho23)
    lconn[2, 1, 2] = fac1 * (-2.0 * r1 + a2sth2) * irho23
    lconn[2, 1, 3] = 0.0
    lconn[2, 1, 4] = -bhspin * sth2 * fac3 * fac1 / (r1 * rho23)

    lconn[2, 2, 1] = lconn[2, 1, 2]
    lconn[2, 2, 2] = (r4 * (-2.0 + r1) * (1.0 + r1) + a2 * (a2 * r1 * (1.0 + 3.0 * r1) * cth4 + a4 * cth4 * cth2 + r3 * sth2 + r1 * cth2 * (2.0 * r1 + 3.0 * r3 - a2sth2))) * irho23
    lconn[2, 2, 3] = -a2 * dthdx2 * s2th / fac2
    lconn[2, 2, 4] = bhspin * sth2 * (a4 * r1 * cth4 + r2 * (2 * r1 + r3 - a2sth2) + a2cth2 * (2.0 * r1 * (-1.0 + r2) + a2sth2)) * irho23

    lconn[2, 3, 1] = lconn[2, 1, 3]
    lconn[2, 3, 2] = lconn[2, 2, 3]
    lconn[2, 3, 3] = -fac3 * dthdx22 * irho2
    lconn[2, 3, 4] = 0.0

    lconn[2, 4, 1] = lconn[2, 1, 4]
    lconn[2, 4, 2] = lconn[2, 2, 4]
    lconn[2, 4, 3] = lconn[2, 3, 4]
    lconn[2, 4, 4] = -fac3 * sth2 * (r1 * rho22 - a2 * fac1 * sth2) / (r1 * rho23)

    lconn[3, 1, 1] = -a2 * r1 * s2th * irho23_dthdx2
    lconn[3, 1, 2] = r1 * lconn[3, 1, 1]
    lconn[3, 1, 3] = 0.0
    lconn[3, 1, 4] = bhspin * r1 * (a2 + r2) * s2th * irho23_dthdx2

    lconn[3, 2, 1] = lconn[3, 1, 2]
    lconn[3, 2, 2] = r2 * lconn[3, 1, 1]
    lconn[3, 2, 3] = r2 * irho2
    lconn[3, 2, 4] = (bhspin * r1 * cth * sth * (r3 * (2.0 + r1) + a2 * (2.0 * r1 * (1.0 + r1) * cth2 + a2 * cth4 + 2 * r1sth2))) * irho23_dthdx2

    lconn[3, 3, 1] = lconn[3, 1, 3]
    lconn[3, 3, 2] = lconn[3, 2, 3]
    lconn[3, 3, 3] = -a2 * cth * sth * dthdx2 * irho2 + d2thdx22 / dthdx2
    lconn[3, 3, 4] = 0.0

    lconn[3, 4, 1] = lconn[3, 1, 4]
    lconn[3, 4, 2] = lconn[3, 2, 4]
    lconn[3, 4, 3] = lconn[3, 3, 4]
    lconn[3, 4, 4] = -cth * sth * (rho23 + a2sth2 * rho2 * (r1 * (4.0 + r1) + a2cth2) + 2.0 * r1 * a4 * sth4) * irho23_dthdx2

    lconn[4, 1, 1] = bhspin * fac1_rho23
    lconn[4, 1, 2] = r1 * lconn[4, 1, 1]
    lconn[4, 1, 3] = -2.0 * bhspin * r1 * cth * dthdx2 / (sth * rho22)
    lconn[4, 1, 4] = -a2sth2 * fac1_rho23

    lconn[4, 2, 1] = lconn[4, 1, 2]
    lconn[4, 2, 2] = bhspin * r2 * fac1_rho23
    lconn[4, 2, 3] = -2 * bhspin * r1 * (a2 + 2 * r1 * (2.0 + r1) + a2 * c2th) * cth * dthdx2 / (sth * fac2 * fac2)
    lconn[4, 2, 4] = r1 * (r1 * rho22 - a2sth2 * fac1) * irho23

    lconn[4, 3, 1] = lconn[4, 1, 3]
    lconn[4, 3, 2] = lconn[4, 2, 3]
    lconn[4, 3, 3] = -bhspin * r1 * dthdx22 * irho2
    lconn[4, 3, 4] = dthdx2 * (0.25 * fac2 * fac2 * cth / sth + a2 * r1 * s2th) * irho22

    lconn[4, 4, 1] = lconn[4, 1, 4]
    lconn[4, 4, 2] = lconn[4, 2, 4]
    lconn[4, 4, 3] = lconn[4, 3, 4]
    lconn[4, 4, 4] = (-bhspin * r1sth2 * rho22 + a3 * sth4 * fac1) * irho23
    return SArray(lconn)
end

"""
    FMKS_connection_analytic(X, bhspin, model)

Compute the analytic connection coefficients for the Funky Modified
Kerr-Schild (FMKS) metric, as used by `Iharm`.

# Arguments
- `X`: Position four-vector in internal coordinates.
- `bhspin`: Dimensionless black hole spin parameter.
- `model`: Model parameters (`model.hslope`, `model.poly_xt`,
  `model.poly_alpha`, `model.poly_norm`, `model.mks_smooth`,
  `model.startx`).

# Returns
- The connection coefficients `Γ^μ_{αβ}`, as an `SArray`.
"""
@inline function FMKS_connection_analytic(X::AbstractVector{T}, bhspin, model) where {T}
    lconn = MArray{Tuple{4,4,4},T,3,64}(undef)

    r1 = exp(X[2])

    drdx1 = r1
    y = 2.0 * X[3] - 1.0
    y_over_xt = y / model.poly_xt
    pow_alpha = y_over_xt^model.poly_alpha

    thG = π * X[3] + ((1.0 - model.hslope) / 2.0) * sin(2.0 * π * X[3])
    thJ = model.poly_norm * y * (1.0 + pow_alpha / (model.poly_alpha + 1.0)) + 0.5 * π
    W = exp(model.mks_smooth * (model.startx[2] - X[2]))
    dWdx1 = -model.mks_smooth * W
    d2Wdx12 = model.mks_smooth * model.mks_smooth * W

    dthGdx2 = π + π * (1.0 - model.hslope) * cos(2.0 * π * X[3])
    d2thGdx22 = -2.0 * π * π * (1.0 - model.hslope) * sin(2.0 * π * X[3])

    dthJdx2 = 2.0 * model.poly_norm * (1.0 + pow_alpha)
    pow_alpha_minus_1 = y_over_xt^(model.poly_alpha - 1.0)
    d2thJdx22 = 4.0 * model.poly_norm * model.poly_alpha / model.poly_xt * pow_alpha_minus_1

    th = thG + W * (thJ - thG)
    dthdx1 = dWdx1 * (thJ - thG)
    dthdx2 = dthGdx2 + W * (dthJdx2 - dthGdx2)
    d2thdx12 = d2Wdx12 * (thJ - thG)
    d2thdx1dx2 = dWdx1 * (dthJdx2 - dthGdx2)
    d2thdx22 = d2thGdx22 + W * (d2thJdx22 - d2thGdx22)

    r2 = r1 * r1
    r3 = r2 * r1
    r4 = r3 * r1

    sth = sin(th)
    cth = cos(th)

    sth2 = sth * sth
    r1sth2 = r1 * sth2
    sth4 = sth2 * sth2
    cth2 = cth * cth
    cth4 = cth2 * cth2
    s2th = 2.0 * sth * cth
    c2th = 2.0 * cth2 - 1.0

    a2 = bhspin * bhspin
    a2sth2 = a2 * sth2
    a2cth2 = a2 * cth2
    a3 = a2 * bhspin
    a4 = a3 * bhspin
    a4cth4 = a4 * cth4

    rho2 = r2 + a2cth2
    rho22 = rho2 * rho2
    rho23 = rho22 * rho2

    irho2 = 1.0 / rho2
    irho22 = irho2 * irho2
    irho23 = irho22 * irho2

    fac1 = r2 - a2cth2
    fac1_rho23 = fac1 * irho23
    fac2 = a2 + 2 * r2 + a2 * c2th
    fac3 = a2 + r1 * (-2.0 + r1)

    gamma002 = -a2 * r1 * s2th * irho22

    lconn[1, 1, 1] = 2.0 * r1 * fac1_rho23
    lconn[1, 1, 2] = drdx1 * (2 * r1 + rho2) * fac1_rho23 + gamma002 * dthdx1
    lconn[1, 1, 3] = gamma002 * dthdx2
    lconn[1, 1, 4] = -2.0 * bhspin * r1sth2 * fac1_rho23

    base_011 = 2.0 * (r4 + r1 * fac1 - a4cth4) * irho23
    drdx1_2 = drdx1 * drdx1
    dthdx1_2 = dthdx1 * dthdx1

    lconn[1, 2, 1] = lconn[1, 1, 2]
    lconn[1, 2, 2] = base_011 * drdx1_2 + 2.0 * gamma002 * drdx1 * dthdx1 - 2.0 * r2 * irho2 * dthdx1_2
    lconn[1, 2, 3] = dthdx2 * (gamma002 * drdx1 - 2.0 * r2 * irho2 * dthdx1)
    lconn[1, 2, 4] = bhspin * drdx1 * (-r1 * (r3 + 2.0 * fac1) + a4cth4) * sth2 * irho23 + a3 * r1sth2 * s2th * irho22 * dthdx1

    dthdx2_2 = dthdx2 * dthdx2
    lconn[1, 3, 1] = lconn[1, 1, 3]
    lconn[1, 3, 2] = lconn[1, 2, 3]
    lconn[1, 3, 3] = -2.0 * r2 * irho2 * dthdx2_2
    lconn[1, 3, 4] = a3 * r1sth2 * s2th * irho22 * dthdx2

    lconn[1, 4, 1] = lconn[1, 1, 4]
    lconn[1, 4, 2] = lconn[1, 2, 4]
    lconn[1, 4, 3] = lconn[1, 3, 4]
    lconn[1, 4, 4] = 2.0 * r1sth2 * (-r1 * rho22 + a2sth2 * fac1) * irho23

    idrdx1 = 1.0 / drdx1
    idthdx2 = 1.0 / dthdx2
    idrdx1_idthdx2 = idthdx2 * idrdx1

    lconn[2, 1, 1] = fac3 * fac1 * irho23 * idrdx1
    lconn[2, 1, 2] = fac1 * (-2.0 * r1 + a2sth2) * irho23
    lconn[2, 1, 3] = 0.0
    lconn[2, 1, 4] = -bhspin * sth2 * lconn[2, 1, 1]

    term_111_1 = -(r2 - a2cth2) * (4.0 * r1 + fac3 - 2.0 * a2sth2) * irho23
    term_111_2 = -2.0 * a2 * s2th / fac2
    term_111_3 = -fac3 * irho2

    dthdx1_2 = dthdx1 * dthdx1

    lconn[2, 2, 1] = lconn[2, 1, 2]
    lconn[2, 2, 2] = term_111_1 * drdx1 + 1.0 + term_111_2 * dthdx1 + term_111_3 * dthdx1_2
    lconn[2, 2, 3] = dthdx2 * (0.5 * term_111_2 + term_111_3 * r1 * idrdx1 * dthdx1)
    lconn[2, 2, 4] = bhspin * sth2 * (a4 * r1 * cth4 + r2 * (2.0 * r1 + r3 - a2sth2) + a2cth2 * (2.0 * r1 * (-1.0 + r2) + a2sth2)) * irho23

    dthdx2_2 = dthdx2 * dthdx2

    lconn[2, 3, 2] = lconn[2, 2, 3]
    lconn[2, 3, 3] = term_111_3 * r1 * idrdx1 * dthdx2_2
    lconn[2, 3, 4] = 0.0
    lconn[2, 4, 1] = lconn[2, 1, 4]
    lconn[2, 4, 2] = lconn[2, 2, 4]
    lconn[2, 4, 3] = lconn[2, 3, 4]
    lconn[2, 4, 4] = -fac3 * sth2 * (r1 * rho22 - a2 * fac1 * sth2) * irho23 * idrdx1

    gamma002_irho2 = gamma002 * irho2

    lconn[3, 1, 1] = gamma002_irho2 * idthdx2 - fac3 * fac1_rho23 * dthdx1 * idrdx1_idthdx2
    lconn[3, 1, 2] = (gamma002_irho2 * drdx1 - lconn[2, 1, 2] * dthdx1) * idthdx2
    lconn[3, 1, 3] = 0.0
    lconn[3, 1, 4] = bhspin * r1 * (a2 + r2) * s2th * irho23 * idthdx2 + bhspin * sth2 * fac3 * fac1 * irho23 * dthdx1 * idrdx1_idthdx2

    drdx1_2 = drdx1 * drdx1
    dthdx1_3 = dthdx1_2 * dthdx1

    lconn[3, 2, 1] = lconn[3, 1, 2]
    lconn[3, 2, 2] = (gamma002_irho2 * drdx1_2 - term_111_1 * drdx1 * dthdx1 + 2.0 * r1 * irho2 * drdx1 * dthdx1 - term_111_2 * dthdx1_2 - a2 * cth * sth * irho2 * dthdx1_2 - term_111_3 * dthdx1_3 - dthdx1 + d2thdx12) * idthdx2

    lconn[3, 2, 3] = r1 * irho2 * drdx1 - 0.5 * term_111_2 * dthdx1 - a2 * cth * sth * irho2 * dthdx1 - term_111_3 * r1 * idrdx1 * dthdx1_2 + d2thdx1dx2 * idthdx2

    term_213_a = bhspin * cth * sth * (r3 * (2.0 + r1) + a2 * (2.0 * r1 * (1.0 + r1) * cth2 + a2 * cth4 + 2.0 * r1sth2)) * irho23

    lconn[3, 2, 4] = (term_213_a * drdx1 - lconn[2, 2, 4] * dthdx1) * idthdx2
    lconn[3, 3, 1] = lconn[3, 1, 3]
    lconn[3, 3, 2] = lconn[3, 2, 3]
    lconn[3, 3, 3] = -a2 * cth * sth * irho2 * dthdx2 + d2thdx22 * idthdx2 - term_111_3 * r1 * idrdx1 * dthdx2 * dthdx1
    lconn[3, 3, 4] = 0.0

    term_233_a = -cth * sth * (rho23 + a2sth2 * rho2 * (r1 * (4.0 + r1) + a2cth2) + 2.0 * r1 * a4 * sth4) * irho23
    lconn[3, 4, 1] = lconn[3, 1, 4]
    lconn[3, 4, 2] = lconn[3, 2, 4]
    lconn[3, 4, 3] = lconn[3, 3, 4]
    lconn[3, 4, 4] = (term_233_a - lconn[2, 4, 4] * dthdx1) * idthdx2

    lconn[4, 1, 1] = bhspin * fac1_rho23
    term_301_b = -2.0 * bhspin * r1 * cth / (sth * rho22)
    lconn[4, 1, 2] = lconn[4, 1, 1] * drdx1 + term_301_b * dthdx1
    lconn[4, 1, 3] = term_301_b * dthdx2
    lconn[4, 1, 4] = -a2sth2 * fac1_rho23

    term_311_2 = -2.0 * bhspin * (a2 + 2.0 * r1 * (2.0 + r1) + a2 * c2th) * cth / (sth * fac2 * fac2)
    term_311_3 = -bhspin * r1 * irho2

    drdx1_2 = drdx1 * drdx1
    dthdx1_2 = dthdx1 * dthdx1

    lconn[4, 2, 1] = lconn[4, 1, 2]
    lconn[4, 2, 2] = lconn[4, 1, 1] * drdx1_2 + 2.0 * term_311_2 * drdx1 * dthdx1 + term_311_3 * dthdx1_2
    lconn[4, 2, 3] = term_311_2 * drdx1 * dthdx2 + term_311_3 * dthdx2 * dthdx1
    term_313_1 = (r1 * rho22 - a2sth2 * fac1) * irho23
    term_313_2 = (0.25 * fac2 * fac2 * cth / sth + a2 * r1 * s2th) * irho22
    lconn[4, 2, 4] = term_313_1 * drdx1 + term_313_2 * dthdx1

    dthdx2_2 = dthdx2 * dthdx2

    lconn[4, 3, 1] = lconn[4, 1, 3]
    lconn[4, 3, 2] = lconn[4, 2, 3]
    lconn[4, 3, 3] = term_311_3 * dthdx2_2
    lconn[4, 3, 4] = term_313_2 * dthdx2

    lconn[4, 4, 1] = lconn[4, 1, 4]
    lconn[4, 4, 2] = lconn[4, 2, 4]
    lconn[4, 4, 3] = lconn[4, 3, 4]
    lconn[4, 4, 4] = (-bhspin * r1sth2 * rho22 + a3 * sth4 * fac1) * irho23

    return SArray(lconn)
end

"""
    get_connection_analytic(X, bhspin, model)

Compute the analytic connection coefficients at `X`, selecting the plain
MKS or FMKS formula based on `model.metric`.

# Arguments
- `X`: Position four-vector in internal coordinates.
- `bhspin`: Dimensionless black hole spin parameter.
- `model`: Model parameters (`model.metric`).

# Returns
- The connection coefficients `Γ^μ_{αβ}`.
"""
@inline function get_connection_analytic(X::AbstractVector{T}, bhspin, model) where {T}
    if model.metric == Metrics.METRIC_MKS
        return Models_and_MKS_connection_analytic(X, bhspin, model)
    else
        return FMKS_connection_analytic(X, bhspin, model)
    end
end

"""
    get_connection_analytic!(X, lconn, bhspin, model)

In-place version of [`get_connection_analytic`](@ref), writing the result
into `lconn`.

# Arguments
- `X`: Position four-vector in internal coordinates.
- `lconn`: Output tensor, overwritten with the connection coefficients.
- `bhspin`: Dimensionless black hole spin parameter.
- `model`: Model parameters (`model.metric`).
"""
@inline function get_connection_analytic!(X::AbstractVector{T}, lconn, bhspin, model) where {T}
    if model.metric == Metrics.METRIC_MKS
        lconn .= Models_and_MKS_connection_analytic(X, bhspin, model)
    else
        lconn .= FMKS_connection_analytic(X, bhspin, model)
    end
end

"""
    compute_dKcon(dl, lconn, Kcon)

Compute the change in photon 4-momentum over a step of size `dl`, given
the connection coefficients `lconn`.

# Arguments
- `dl`: Step size along the geodesic.
- `lconn`: Connection coefficients `Γ^μ_{αβ}`.
- `Kcon`: Contravariant photon 4-momentum.

# Returns
- The change in 4-momentum, `dKcon`.
"""
@inline function compute_dKcon(dl::Float64, lconn, Kcon::SVector{4,Float64})
    dK1 = dK2 = dK3 = dK4 = 0.0
    @inbounds for i in 1:4, j in 1:4
        term = Kcon[i] * Kcon[j] * dl
        dK1 -= lconn[1, i, j] * term
        dK2 -= lconn[2, i, j] * term
        dK3 -= lconn[3, i, j] * term
        dK4 -= lconn[4, i, j] * term
    end
    return SVector{4,Float64}(dK1, dK2, dK3, dK4)
end

"""
    push_photon(X, Kcon, dl, lconn, bhspin, model)

Advance the photon position and 4-momentum by a step `dl` (forward or
backward, depending on its sign), using a midpoint (RK2) integrator.

# Arguments
- `X`: Position four-vector in internal coordinates.
- `Kcon`: Contravariant photon 4-momentum.
- `dl`: Step size along the geodesic.
- `lconn`: Scratch tensor for the connection coefficients.
- `bhspin`: Dimensionless black hole spin parameter.
- `model`: Model parameters.

# Returns
- A tuple `(new_X, new_Kcon, Xhalf, Kconhalf)`.
"""
Base.@inline function push_photon(X::SVector{4,Float64}, Kcon::SVector{4,Float64}, dl::Float64, bhspin::Float64, model)
    lconn_half = get_connection_analytic(X, bhspin, model)
    dKcon_half = compute_dKcon(0.5 * dl, lconn_half, Kcon)
    Kconhalf = Kcon + dKcon_half
    Xhalf = X + (0.5 * dl) * Kcon

    lconn_full = get_connection_analytic(Xhalf, bhspin, model)
    dKcon_full = compute_dKcon(dl, lconn_full, Kconhalf)
    new_Kcon = Kcon + dKcon_full
    new_X = X + dl * Kconhalf

    return new_X, new_Kcon, Xhalf, Kconhalf
end

"""
    push_photon!(X, Kcon, dl, Xhalf, Kconhalf, lconn, bhspin, model)

In-place version of [`push_photon`](@ref).

# Arguments
- `X`: Position four-vector, overwritten with the new position.
- `Kcon`: Contravariant photon 4-momentum, overwritten with the new value.
- `dl`: Step size along the geodesic.
- `Xhalf`: Output vector, overwritten with the half-step position.
- `Kconhalf`: Output vector, overwritten with the half-step 4-momentum.
- `lconn`: Scratch tensor for the connection coefficients.
- `bhspin`: Dimensionless black hole spin parameter.
- `model`: Model parameters.
"""
Base.@inline function push_photon!(X::MVector{4,Float64}, Kcon::MVector{4,Float64}, dl::Float64, Xhalf::MVector{4,Float64}, Kconhalf::MVector{4,Float64}, lconn, bhspin::Float64, model)
    dKcon::Float64 = 0.0

    get_connection_analytic!(X, lconn, bhspin, model)

    @inbounds for k in 1:Constants.NDIM
        @inbounds for i in 1:Constants.NDIM
            @inbounds for j in 1:Constants.NDIM
                dKcon -= 0.5 * dl * lconn[k, i, j] * Kcon[i] * Kcon[j]
            end
        end
        Kconhalf[k] = Kcon[k] + dKcon
        Xhalf[k] = X[k] + 0.5 * dl * Kcon[k]
        dKcon = 0.0
    end

    get_connection_analytic!(Xhalf, lconn, bhspin, model)

    dKcon = 0.0

    @inbounds for k in 1:Constants.NDIM
        @inbounds for i in 1:Constants.NDIM
            @inbounds for j in 1:Constants.NDIM
                dKcon -= dl * lconn[k, i, j] * Kconhalf[i] * Kconhalf[j]
            end
        end
        Kcon[k] += dKcon
        X[k] += dl * Kconhalf[k]
        dKcon = 0.0
    end
end

const DEL = 1.e-6

"""
    get_connection(X, bhspin, model, conn)

Compute the connection coefficients at `X` via finite differences of the
metric tensor (an alternative to the closed-form
[`get_connection_analytic`](@ref)).

# Arguments
- `X`: Position four-vector in internal coordinates.
- `bhspin`: Dimensionless black hole spin parameter.
- `model`: Model parameters.
- `conn`: Output tensor, overwritten with the connection coefficients.

# Returns
- `conn`.
"""
function get_connection(X::AbstractVector{T}, bhspin, model, conn) where {T}
    tmp = similar(conn)
    Xh = copy(X)
    Xl = copy(X)

    gcov = Metrics.gcov_func(X, bhspin, model)
    gcon = Metrics.gcon_func(gcov)

    @inbounds for k in 1:Constants.NDIM
        Xh .= X
        Xl .= X
        Xh[k] += DEL
        Xl[k] -= DEL

        gh = Metrics.gcov_func(Xh, bhspin, model)
        gl = Metrics.gcov_func(Xl, bhspin, model)

        @inbounds for i in 1:Constants.NDIM
            @inbounds for j in 1:Constants.NDIM
                conn[i, j, k] = (gh[i, j] - gl[i, j]) / (Xh[k] - Xl[k])
            end
        end
    end

    @inbounds for i in 1:Constants.NDIM
        @inbounds for j in 1:Constants.NDIM
            @inbounds for k in 1:Constants.NDIM
                tmp[i, j, k] = 0.5 * (conn[j, i, k] + conn[k, i, j] - conn[k, j, i])
            end
        end
    end

    @inbounds for i in 1:Constants.NDIM
        @inbounds for j in 1:Constants.NDIM
            @inbounds for k in 1:Constants.NDIM
                s = 0.0
                for l in 1:Constants.NDIM
                    s += gcon[i, l] * tmp[l, j, k]
                end
                conn[i, j, k] = s
            end
        end
    end
    return conn
end

"""
    stepsize(X, Kcon, cstartx, cstopx, eps_ipole=0.01)

Compute the adaptive step size for the geodesic integration.

# Arguments
- `X`: Position four-vector in internal coordinates.
- `Kcon`: Contravariant photon 4-momentum.
- `cstartx`: Lower bound of the native coordinate grid.
- `cstopx`: Upper bound of the native coordinate grid.
- `eps_ipole`: Step-size control parameter.

# Returns
- The step size `dl`.
"""
function stepsize(X, Kcon, cstartx, cstopx, eps_ipole::Float64=0.01)
 
    if (true)
        deh::Float64 = min(abs(X[2] - cstartx[2]), 0.1)
        dlx2 = eps_ipole * (10 * deh) / (abs(Kcon[2]) + Constants.SMALL * Constants.SMALL)
        cut::Float64 = 0.02
        lx3::Float64 = cstopx[3] - cstartx[3]
        dpole::Float64 = min(abs(X[3] / lx3), abs((cstopx[3] - X[3]) / lx3))
        d2fac::Float64 = (dpole < cut) ? dpole / 3 : min(cut / 3 + (dpole - cut) * 10.0, 1)
        dlx3 = eps_ipole * d2fac / (abs(Kcon[3]) + Constants.SMALL * Constants.SMALL)
    else
        dlx2 = eps_ipole / (abs(Kcon[2]) + Constants.SMALL*Constants.SMALL)
        dlx3 = eps_ipole * min(X[3], 1. - X[3]) / (abs(Kcon[3]) + Constants.SMALL*Constants.SMALL)
    end
    dlx4 = eps_ipole / (abs(Kcon[4]) + Constants.SMALL * Constants.SMALL)
    idlx2 = 1.0 / (abs(dlx2) + Constants.SMALL*Constants.SMALL)
    idlx3 = 1.0 / (abs(dlx3) + Constants.SMALL*Constants.SMALL)
    idlx4 = 1.0 / (abs(dlx4) + Constants.SMALL*Constants.SMALL)
    return 1.0 / (idlx2 + idlx3 + idlx4)
end

"""
    stop_backward_integration(X, Kcon, Rh, Rstop)

Check whether the backward geodesic integration should stop.

# Arguments
- `X`: Position four-vector in internal coordinates.
- `Kcon`: Contravariant photon 4-momentum.
- `Rh`: Event horizon radius.
- `Rstop`: Backward-integration stopping radius.

# Returns
- `1` if the integration should stop, `0` otherwise.
"""
function stop_backward_integration(X, Kcon, Rh::Float64, Rstop::Float64)
    if ((X[2] > log(Rstop)) && (Kcon[2] < 0.0)) || (X[2] < log(Rh + 0.0001))
        return 1
    end

    return 0
end

"""
    trace_geodesic(Xi, Kconi, traj, i, j, bhspin, Rh, Rstop, model)

Trace the photon geodesic backward from the camera, starting at position
`Xi` with 4-momentum `Kconi`, filling `traj` with the trajectory steps.

# Arguments
- `Xi`: Initial position four-vector, in internal coordinates.
- `Kconi`: Initial contravariant photon 4-momentum.
- `traj`: Output trajectory vector.
- `i`, `j`: Pixel indices in the image plane.
- `bhspin`: Dimensionless black hole spin parameter.
- `Rh`: Event horizon radius.
- `Rstop`: Backward-integration stopping radius.
- `model`: Model parameters (`model.Rout`, `model.L_unit`,
  `model.cstartx`, `model.cstopx`).

# Returns
- A tuple `(nstep, midplane_crossings)`.
"""
function trace_geodesic(Xi::SVector{4,Float64}, Kconi::SVector{4,Float64}, traj::Vector{OfTrajS}, i::Int, j::Int, bhspin::Float64, Rh::Float64, Rstop::Float64, model)
    X = Xi
    Kcon = Kconi

    traj[1] = OfTrajS(0.0, Xi, Kconi, Xi, Kconi)

    midplane_crossings = 0
    _, th = Coordinates.bl_coord(Xi, model)
    position_in_midplane = th > π / 2 ? 1 : 0

    nstep = 1
    #lconn = MArray{Tuple{4,4,4},Float64,3,64}(undef)
    ABSOLUTE_MAX = 50000

    while (stop_backward_integration(X, Kcon, Rh, Rstop) == 0) && (nstep < ABSOLUTE_MAX)
        if nstep >= length(traj)
            @warn "Trajectory array length exceeded at pixel ($i, $j)."
            resize!(traj, length(traj) * 2)

            dummy = @SVector zeros(4)
            for k in (nstep+1):length(traj)
                traj[k] = OfTrajS(0.0, dummy, dummy, dummy, dummy)
            end
        end

        dl = stepsize(X, Kcon, model.cstartx, model.cstopx)
        unit_dl = dl * model.L_unit * Constants.HPL / (Constants.ME * Constants.CL^2)

        traj[nstep] = OfTrajS(unit_dl, traj[nstep].X, traj[nstep].Kcon, traj[nstep].Xhalf, traj[nstep].Kconhalf)

        _, th = Coordinates.bl_coord(X, model)
        if (position_in_midplane == 1) && (th <= π / 2)
            position_in_midplane = 0
            midplane_crossings += 1
        elseif (position_in_midplane == 0) && (th > π / 2)
            position_in_midplane = 1
            midplane_crossings += 1
        end

        new_X, new_Kcon, Xhalf, Kconhalf = push_photon(X, Kcon, -dl, bhspin, model)

        nstep += 1

        traj[nstep] = OfTrajS(0.0, new_X, new_Kcon, Xhalf, Kconhalf)

        X = new_X
        Kcon = new_Kcon
    end

    return nstep, midplane_crossings
end

end
