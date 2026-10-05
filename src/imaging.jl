"""
Top-level image-plane orchestration: batch intensity integration and
image/flux reporting.
"""
module Imaging

using ForwardDiff
using Printf
using StaticArrays
using ProgressMeter
using ..Constants
using ..GeoTypes
using ..Camera
using ..Geodesics
using ..Radiation
using ..Tetrads

export output_stokes_parameters, calculate_scale_factor, raytrace_image_gpu!, raytrace_image,  render_image_gpu!

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
        Xcam, Econ, bhspin, nx, ny, nmaxstep, freq, fovx, fovy, Rout, Rstop, data, params)

GPU kernel that raytraces one tile of the image plane, launch it with `@cuda`.
Its method lives in the `JipoleCUDAExt` extension, so `using CUDA` must come first.
"""
function raytrace_image_gpu! end

"""
    render_image_gpu!(Image, model, gpu_sim_data, ro, θo, phi, freq, fovx, fovy, nx, ny;
        nmaxstep=16000, nmaxstep_ceiling=50000, block_size=64)

Raytrace the full image on the GPU tile by tile into the host array `Image`. A tile with a
truncated geodesic is re-run with `nmaxstep` doubled, up to `nmaxstep_ceiling`. Returns the
`nmaxstep` finally used. Its method lives in the `JipoleCUDAExt` extension.
"""
function render_image_gpu! end

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
function raytrace_image(model, simulation_data, ro, th, phi, freq, pixels_x, pixels_y,
                         fovx, fovy, maxnstep, Rh, xoff, yoff)
    T = promote_type(typeof(ro), typeof(th), typeof(phi), typeof(model.a))

    Xcamera = SVector{4,T}(Camera.camera_position(ro, th, phi, model.a, model))
    _, Econ, _ = Tetrads.make_camera_tetrad(Xcamera, model.a, model)
    freq_unitless = freq * Constants.HPL / (Constants.ME * Constants.CL * Constants.CL)

    Image = Matrix{T}(undef, pixels_x, pixels_y)

    println("Allocating workspaces for $pixels_x row-tasks...")
    nbuf = Threads.nthreads()
    traj_pool = Channel{Vector{GeoTypes.OfTrajGeneric{T}}}(nbuf)
    for _ in 1:nbuf
        put!(traj_pool, Vector{GeoTypes.OfTrajGeneric{T}}(undef, maxnstep))
    end

    p = Progress(pixels_x * pixels_y; desc = "Raytracing Image...", showspeed = true, barlen = 30)
    ProgressMeter.ijulia_behavior(:clear)
    progress_lock = ReentrantLock()
    println("Tracing Geodesics...")
    Threads.@threads :greedy for i in 0:(pixels_x - 1)
        my_traj = take!(traj_pool)
        try
            for j in 0:(pixels_y - 1)
                nstep, _ = Geodesics.get_pixel(
                    my_traj, i, j, Xcamera, Econ,
                    fovx, fovy, freq_unitless,
                    pixels_x, pixels_y, model.a,
                    Rh, model.rmax_geo, model, xoff, yoff
                )

                Radiation.integrate_emission!(
                    my_traj, nstep, Image,
                    i + 1, j + 1, freq, model.a, model, simulation_data
                )
            end
            lock(progress_lock) do
                ProgressMeter.next!(p; step = pixels_y)
            end
        finally
            put!(traj_pool, my_traj)
        end
    end
    Image .*= freq^3
    finish!(p)

    return Image
end


end
