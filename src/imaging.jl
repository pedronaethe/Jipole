"""
Top-level image-plane orchestration: batch intensity integration and
image/flux reporting.
"""
module Imaging

using ForwardDiff
using CUDA
using Printf
using StaticArrays
using ProgressMeter
using ..Constants
using ..GeoTypes
using ..Camera
using ..Geodesics
using ..Radiation

export output_stokes_parameters, calculate_scale_factor

"""
    output_stokes_parameters(Image, freq_cgs, scale_factor, res, Dsource)

Print the total flux, average and peak intensity, and `nuLnu` for the
computed image.

# Arguments
- `Image`: Computed intensity image.
- `freq_cgs`: Frequency, in cgs units.
- `scale_factor`: Scale factor for the image intensity (see
  [`calculate_scale_factor`](@ref)).
- `nx`: Image resolution in the x-direction.
- `ny`: Image resolution in the y-direction.
- `Dsource`: Distance to the source, in cm.
"""
function output_stokes_parameters(Image, freq_cgs, scale_factor, nx, ny, Dsource)
    println("Image processing complete. Calculating total flux and averages...")
    Ftot::Float64 = 0.0
    Iavg::Float64 = 0.0
    Imax::Float64 = 0.0
    imax::Int = 0
    jmax::Int = 0
    for i in 1:nx
        for j in 1:ny
            Ftot += Image[i, j] * scale_factor
            Iavg += Image[i, j]
            if (Image[i, j]) > Imax
                imax = i
                jmax = j
                Imax = Image[i, j]
            end
        end
    end
    Iavg *= 1.0 / (nx * ny)
    println("Scale = $scale_factor")
    println("imax = $imax, jmax = $jmax, Imax = $Imax, Iavg = $Iavg")
    println("Total Flux Fnu = $Ftot Jy")
    println("nuLnu = $(Ftot * Dsource * Dsource * Constants.JY * freq_cgs * 4.0 * π)")
end

"""
    calculate_scale_factor(sizex, sizey, pixelsx, pixelsy, SourceD, LengthUnit)

Compute the scale factor for the image, converting the per-pixel
intensity to a flux density in Jankys (JY).

# Arguments
- `sizex`, `sizey`: Image size, in `LengthUnit`.
- `nx`, `ny`: Image resolution in the x-direction and y-direction, respectively.
- `SourceD`: Distance to the source, in cm.
- `LengthUnit`: Length unit, in cm (e.g. `model.L_unit`).

# Returns
- The scale factor.
"""
@inline function calculate_scale_factor(sizex, sizey, nx, ny, SourceD, LengthUnit)
    return (sizex * LengthUnit / nx) * (sizey * LengthUnit / ny) / (SourceD * SourceD) / Constants.JY
end


"""
    raytrace_image_gpu!(d_traj, d_Image, d_truncated, i_offset, j_offset, block_size_x, block_size_y,
        ro, θo, phi, bhspin, nx, ny, nmaxstep, freq, fovx, fovy, Rout, Rstop, data, params)

GPU kernel launcher for [`calculate_image!`](@ref): raytraces and
integrates the emission for one tile of the image plane (the tile given by
`i_offset`/`j_offset`, of size `block_size_x`×`block_size_y`), writing the
resulting intensities into `d_Image`. Call once per tile from a
`for`-loop over the full image, since `d_traj`'s size limits how much of
the image a single launch can cover.

The division in tiles is necessary depending on the size of the image due to GPU memory constraints.

# Arguments
- `d_traj`: Pre-allocated `CuArray{OfTrajGRMHD}` scratch buffer, sized
  `(block_size_x, block_size_y, nmaxstep)`.
- `d_Image`: Output image, overwritten in-place.
- `d_truncated`: Pre-allocated array of booleans, set `true` for any pixel whose geodesic
  integration hit `nmaxstep` before `stop_backward_integration` actually
  fired (i.e. `d_traj` was too small for that pixel). Callers should check
  this after the launch and retry with a larger `nmaxstep`/`d_traj` if any
  entry came back `true`.
- `i_offset`, `j_offset`: Pixel offset of this tile within the full image.
- `block_size_x`, `block_size_y`: Tile size (must match `d_traj`'s first
  two dimensions and the launch's thread/block configuration).
- `ro`, `θo`, `phi`: Camera radial distance, inclination, and azimuth.
- `bhspin`: Dimensionless black hole spin parameter.
- `nx`, `ny`: Full image resolution.
- `nmaxstep`: Maximum number of geodesic integration steps.
- `freq`: Pivotal frequency, in cgs units.
- `fovx`, `fovy`: Field of view, in radians.
- `Rout`: Outer simulation-grid radius.
- `Rstop`: Backward-integration stopping radius.
- `data`: Model-specific auxiliary data (e.g. `Iharm`'s GRMHD snapshots).
- `params`: Model parameters.
"""
function raytrace_image_gpu!(
    d_traj, d_Image, d_truncated,
    i_offset, j_offset, block_size_x, block_size_y, # New offset parameters
    ro, θo, phi, bhspin, nx, ny, nmaxstep,
    freq, fovx, fovy, Rout, Rstop, data, params
)
    #TODO (PNM): This function is necessarily GPU specific, maybe it shouldn't be and we can find a way to abstract it out.
    local_i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    local_j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    i = local_i - 1 + i_offset
    j = local_j - 1 + j_offset

    if local_i <= block_size_x && local_j <= block_size_y && i < nx && j < ny

        calculate_image!(
            d_traj, d_Image, d_truncated, ro, θo, phi, bhspin, nx, ny, nmaxstep,
            i, j, local_i, local_j, freq, fovx, fovy, Rout, Rstop, params, data
        )
    end
    return nothing
end


"""
    calculate_image!(traj, d_Image, d_truncated, ro, θo, phi, bhspin, nx, ny, nmaxstep, i_global,
        j_global, i_local, j_local, freq, fovx, fovy, Rout, Rstop, params, data=nothing)

GPU per-pixel kernel body for [`raytrace_image_gpu!`](@ref): traces the
geodesic for pixel `(i_global, j_global)` backward from the camera,
storing each step into `traj[i_local, j_local, :]`, then integrates the
radiative transfer equation forward along the stored trajectory (from the
far end back to the camera), writing the result into `d_Image`.

# Arguments
- `traj`: Pre-allocated `CuDeviceArray{OfTrajGRMHD}` scratch buffer for
  this tile, indexed by the pixel's local (within-tile) coordinates.
- `d_Image`: Output image, overwritten in-place at `(i_global, j_global)`.
- `d_truncated`: Pre-allocated boolean array for this tile, set
  `true` at `(i_local, j_local)` if `nmaxstep` was hit before the geodesic
  actually reached its stopping condition.
- `ro`, `θo`, `phi`: Camera radial distance, inclination, and azimuth.
- `bhspin`: Dimensionless black hole spin parameter.
- `nx`, `ny`: Full image resolution.
- `nmaxstep`: Maximum number of geodesic integration steps.
- `i_global`, `j_global`: Pixel indices in the full image plane.
- `i_local`, `j_local`: Pixel indices within this tile's `traj` buffer.
- `freq`: Pivotal frequency, in cgs units.
- `fovx`, `fovy`: Field of view, in radians.
- `Rout`: Outer simulation-grid radius (unused; kept for call-signature
  symmetry with [`raytrace_image_gpu!`](@ref)).
- `Rstop`: Backward-integration stopping radius.
- `params`: Model parameters.
- `data`: Model-specific auxiliary data (e.g. `Iharm`'s GRMHD snapshots).
"""
function calculate_image!(
    traj, d_Image, d_truncated,
    ro::Float64, θo::Float64, phi::Float64, bhspin::Float64,
    nx::Int64, ny::Int64, nmaxstep::Int64,
    i_global::Int64, j_global::Int64,
    i_local::Int64, j_local::Int64, # Accept local matrix indices directly
    freq::Float64, fovx::Float64, fovy::Float64, Rout::Float64, Rstop::Float64, params,
    data::T_data = nothing
) where {T_data}

    if (i_global >= nx || j_global >= ny)
        return nothing
    end
    Xcam = Camera.camera_position(ro, θo, phi, bhspin, params)

    Kcon0 = Geodesics.init_kcon(i_global, j_global, Xcam, nx, ny, fovx, fovy, bhspin, params)
    Kcon = Kcon0 * (freq * Constants.HPL / (Constants.ME * Constants.CL * Constants.CL))

    dl_unit::Float64 = params.L_unit * Constants.HPL / (Constants.ME * Constants.CL^2)
    Rh = 1.0 + sqrt(1.0 - bhspin * bhspin)

    X = Xcam
    K = Kcon
    Xhalf = SVector{4, Float64}(0.0, 0.0, 0.0, 0.0)
    Khalf = SVector{4, Float64}(0.0, 0.0, 0.0, 0.0)
    zero_vec = SVector{4, Float64}(0.0, 0.0, 0.0, 0.0)
    #lconn = MArray{Tuple{4,4,4},Float64,3,64}(undef)

    step::Int64 = 1
    @inbounds traj[i_local, j_local, step] = OfTrajGRMHD(
        0.0, X, K, Xhalf, Khalf,
        zero_vec, zero_vec
    )
    while (Geodesics.stop_backward_integration(X, K, Rh, Rstop) == 0 && (step < nmaxstep))
        @inbounds begin
            dl = Geodesics.stepsize(X, K, params.cstartx, params.cstopx)
            scaled_dl = dl * dl_unit
            X, K, Xhalf, Khalf = Geodesics.push_photon(X, K, -dl, bhspin, params)
            step += 1
            traj[i_local, j_local, step] = OfTrajGRMHD(
                scaled_dl, X, K, Xhalf, Khalf,
                zero_vec, zero_vec
            )
        end
    end

    @inbounds d_truncated[i_local, j_local] = (step >= nmaxstep) && (Geodesics.stop_backward_integration(X, K, Rh, Rstop) == 0)

    # #Radiative Transfer Integration:

    Intensity = 0.0

    @inbounds Xi_S = traj[i_local, j_local, step].X
    @inbounds Kconi_S = traj[i_local, j_local, step].Kcon

    ji, ki = Radiation.get_jk(Xi_S, Kconi_S, freq, bhspin, params, data)

    @inbounds for nstep in step:-1:2
        Xi_S = traj[i_local, j_local, nstep].X
        Xf_S = traj[i_local, j_local, nstep - 1].X
        Kconi_S = traj[i_local, j_local, nstep].Kcon
        Kconf_S = traj[i_local, j_local, nstep - 1].Kcon
        dl_step = traj[i_local, j_local, nstep].dl

        if !Radiation.radiating_region(Xf_S, params, Rh)
            continue
        end
        jf, kf = Radiation.get_jk(Xf_S, Kconf_S, freq, bhspin, params, data)

        Intensity = Radiation.approximate_solve(Intensity, ji, ki, jf, kf, dl_step)

        CUDA.@cuassert !(isnan(Intensity) || isinf(Intensity)) "NaN/Inf Intensity encountered!"

        ji = jf
        ki = kf
    end

    @inbounds d_Image[i_global + 1, j_global + 1] = Intensity * (freq^3)

    return nothing
end



"""
    raytrace_image(model, simulation_data, Xcamera, freq, pixels_x, pixels_y,
                   fovx, fovy, maxnstep, Rh, xoff, yoff)

Trace rays to form an image of the source.

# Arguments
 - `model`: The radiative transfer model parameters.
 - `simulation_data`: The GRMHD simulation data.
 - `Xcamera`: Camera position in native coordinates.
 - `freq`: Observation frequency.
 - `pixels_x`: Number of pixels in the x-direction.
 - `pixels_y`: Number of pixels in the y-direction.
 - `fovx`: Field of view in the x-direction.
 - `fovy`: Field of view in the y-direction.
 - `maxnstep`: Maximum number of steps for geodesic integration.
 - `Rh`: Schwarzschild radius of the black hole.
 - `xoff`: X-offset for the image.
 - `yoff": Y-offset for the image.

 # Returns
 - Returns: The formed image as a 2D array of Float64 values.

"""
#TODO (PNM): Put the camera parameters inside a camera structure, it will be more neat and organized.
function raytrace_image(model, simulation_data, Xcamera::AbstractVector{T}, freq, pixels_x, pixels_y,
                         fovx, fovy, maxnstep, Rh, xoff, yoff) where {T}
    freq_unitless = freq * Constants.HPL / (Constants.ME * Constants.CL * Constants.CL)

    Image = Matrix{T}(undef, pixels_x, pixels_y)

    println("Allocating workspaces for $pixels_x row-tasks...")
    task_trajs = [Vector{GeoTypes.OfTrajGeneric{T}}(undef, maxnstep) for _ in 1:pixels_x]

    p = Progress(pixels_x * pixels_y; desc = "Raytracing Image...", showspeed = true, barlen = 30)
    ProgressMeter.ijulia_behavior(:clear)
    progress_lock = ReentrantLock()
    println("Tracing Geodesics...")
    Threads.@threads :greedy for i in 0:(pixels_x - 1)
        my_traj = task_trajs[i + 1]

        for j in 0:(pixels_y - 1)
            nstep, _ = Geodesics.get_pixel(
                my_traj, i, j, Xcamera,
                fovx, fovy, freq_unitless,
                pixels_x, pixels_y, model.a,
                Rh, model.rmax_geo, model, xoff, yoff
            )

            Radiation.integrate_emission!(
                my_traj, nstep, Image,
                i + 1, j + 1, freq, model.a, model, simulation_data
            )

            lock(progress_lock) do
                if (i * pixels_y + j) % 2000 == 0
                    ProgressMeter.next!(p; showvalues = [(:pixel, "($i, $j)"), (:total_done, "$(i*pixels_y + j)/$(pixels_x * pixels_y)")])
                else
                    ProgressMeter.next!(p)
                end
            end
        end
    end
    Image *= freq^3
    finish!(p)

    return Image
end
end
