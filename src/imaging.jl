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

export output_stokes_parameters, calculate_scale_factor, raytrace_image_gpu!, raytrace_image, calculate_gradients,  render_image_gpu!, trace_all_geodesics

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
    Image .*= freq^3
    finish!(p)

    return Image
end


"""
    trace_all_geodesics(model, ro, th, phi, freq, pixels_x, pixels_y, fovx, fovy,
                        maxnstep, Rh, xoff, yoff; progress = true)

Trace every pixel's geodesic once and keep it.

Returns `(all_geodesics, nsteps, midplane_crossings, Xcamera, bounds)` where
`all_geodesics[i,j]` is that pixel's trajectory truncated to its own length,
`midplane_crossings[i,j]` is its equatorial-crossing count (0 captured, 1
direct, 2 first indirect, ...), and `bounds` is
`(t_last, tgeoi, tgeof)` — the three observer-time limits the time-dependent
modes schedule against, in geometric time (all negative):

- `t_last`: the most negative time any ray reaches, i.e. the full depth of the
  trajectory set.
- `tgeoi`: the *newest* time any ray still needs, the latest entry into
  `r < rmax_geo`.
- `tgeof`: the *oldest* time any ray needs, the earliest exit through
  `r > rmax_geo` on an outgoing leg.

`Imaging.raytrace_image` fuses tracing with transfer and discards both the
crossing count and the trajectory, which is right for fast light on a single
snapshot. Every time-dependent mode instead needs the trajectories kept — they
are reused for every frame, and for every `p` of a sweep — and brisk light needs
the crossing count to assign lensing bands. This function is that shared step;
it was previously duplicated inline in `scripts/main.jl`'s slow-light branch.

Memory is the thing to watch: the returned set is roughly
`pixels_x * pixels_y * mean(nsteps) * sizeof(OfTrajGeneric{T})` and dominates a
time-dependent run's footprint.
"""
function trace_all_geodesics(model, ro, th, phi, freq, pixels_x::Int, pixels_y::Int,
                             fovx, fovy, maxnstep::Int, Rh, xoff = 0.0, yoff = 0.0;
                             progress::Bool = true)

    T = promote_type(typeof(ro), typeof(th), typeof(phi), typeof(model.a))
    Xcamera = SVector{4,T}(Camera.camera_position(ro, th, phi, model.a, model))
    freq_unitless = freq * Constants.HPL / (Constants.ME * Constants.CL * Constants.CL)

    all_geodesics = Matrix{Vector{GeoTypes.OfTrajGeneric{T}}}(undef, pixels_x, pixels_y)
    nsteps = zeros(Int, pixels_x, pixels_y)
    midplane_crossings = zeros(Int, pixels_x, pixels_y)

    # One scratch buffer per row, reused down the row; `:greedy` hands each row
    # to exactly one task, which is what makes that safe.
    task_trajs = [Vector{GeoTypes.OfTrajGeneric{T}}(undef, maxnstep) for _ in 1:pixels_x]

    row_last = zeros(T, pixels_x)
    row_tgeoi = fill(T(-Inf), pixels_x)
    row_tgeof = zeros(T, pixels_x)
    log_rmax = log(model.rmax_geo)

    pbar = progress ? Progress(pixels_x; desc = "Tracing geodesics ", showspeed = true, barlen = 30) : nothing
    plock = ReentrantLock()

    Threads.@threads :greedy for i in 0:(pixels_x - 1)
        my_traj = task_trajs[i + 1]

        for j in 0:(pixels_y - 1)
            nstep, crossings = Geodesics.get_pixel(
                my_traj, i, j, Xcamera, fovx, fovy, freq_unitless,
                pixels_x, pixels_y, model.a, Rh, model.rmax_geo, model, xoff, yoff)

            nsteps[i + 1, j + 1] = nstep
            midplane_crossings[i + 1, j + 1] = crossings
            all_geodesics[i + 1, j + 1] = my_traj[1:nstep]

            final_t = my_traj[nstep].X[1]
            final_t < row_last[i + 1] && (row_last[i + 1] = final_t)

            # Latest entry into, and earliest outgoing exit from, the emitting
            # radius. Scanning from the camera outward, the first step that
            # satisfies each test is the extremal one, so both are latched on
            # their first hit (the `> 0` sentinel marks "not yet set").
            pixel_tgeoi = one(T)
            pixel_tgeof = one(T)
            for k in 1:nstep
                X = my_traj[k].X
                K = my_traj[k].Kcon
                if pixel_tgeoi > 0 && X[2] < log_rmax
                    pixel_tgeoi = X[1]
                end
                if pixel_tgeof > 0 && X[2] > log_rmax && K[2] < 0
                    pixel_tgeof = X[1]
                end
            end
            pixel_tgeoi < 0 && pixel_tgeoi > row_tgeoi[i + 1] && (row_tgeoi[i + 1] = pixel_tgeoi)
            if pixel_tgeof < 0
                pixel_tgeof < row_tgeof[i + 1] && (row_tgeof[i + 1] = pixel_tgeof)
            elseif final_t < row_tgeof[i + 1]
                row_tgeof[i + 1] = final_t
            end
        end

        if pbar !== nothing
            lock(plock) do
                ProgressMeter.next!(pbar)
            end
        end
    end
    pbar === nothing || finish!(pbar)

    bounds = (t_last = minimum(row_last),
              tgeoi = maximum(row_tgeoi),
              tgeof = minimum(row_tgeof))
    return all_geodesics, nsteps, midplane_crossings, Xcamera, bounds
end

end
