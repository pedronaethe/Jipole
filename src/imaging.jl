"""
Top-level image-plane orchestration: batch intensity integration and
image/flux reporting.
"""
module Imaging

using CUDA
using Printf
using StaticArrays
using ..Constants
using ..GeoTypes
using ..Camera
using ..Geodesics
using ..Radiation

export IpoleGeoIntensityIntegration, OutputStokesParameters, CalculateScaleFactor

"""
    IpoleGeoIntensityIntegration(traj, freq_cgs, nx, ny, bhspin, model, data=nothing)

Integrate the emission along the geodesic trajectory of every pixel.

# Arguments
- `traj`: Matrix of geodesic trajectories, one per pixel.
- `freq_cgs`: Frequency, in cgs units.
- `nx`, `ny`: Image resolution.
- `bhspin`: Dimensionless black hole spin parameter.
- `model`: Model parameters.
- `data`: Model-specific auxiliary data.

# Returns
- The integrated intensity image.
"""
function IpoleGeoIntensityIntegration(traj, freq_cgs, nx::Int, ny::Int, bhspin, model, data=nothing)
    Image = zeros(Float64, nx, ny)
    Threads.@threads for i in 0:(nx-1)
        for j in 0:(ny-1)
            Radiation.integrate_emission!(traj[i+1, j+1], length(traj[i+1, j+1]), Image, i + 1, j + 1, freq_cgs, bhspin, model, data)
        end
    end

    return (Image * freq_cgs^3)
end

"""
    OutputStokesParameters(Image, freq_cgs, scale_factor, res, Dsource)

Print the total flux, average and peak intensity, and `nuLnu` for the
computed image.

# Arguments
- `Image`: Computed intensity image.
- `freq_cgs`: Frequency, in cgs units.
- `scale_factor`: Scale factor for the image intensity (see
  [`CalculateScaleFactor`](@ref)).
- `res`: Image resolution (assumed square).
- `Dsource`: Distance to the source, in cm.
"""
function OutputStokesParameters(Image, freq_cgs, scale_factor, res, Dsource)
    println("Image processing complete. Calculating total flux and averages...")
    Ftot::Float64 = 0.0
    Iavg::Float64 = 0.0
    Imax::Float64 = 0.0
    imax::Int = 0
    jmax::Int = 0
    for i in 1:res
        for j in 1:res
            Ftot += Image[i, j] * scale_factor
            Iavg += Image[i, j]
            if (Image[i, j]) > Imax
                imax = i
                jmax = j
                Imax = Image[i, j]
            end
        end
    end
    Iavg *= 1.0 / (res * res)
    @printf("Scale = %.15e\n", scale_factor)
    println("imax = $imax, jmax = $jmax, Imax = $Imax, Iavg = $Iavg")
    @printf("Total Flux Fnu = %.15e Jy\n", Ftot)
    println("nuLnu = $(Ftot * Dsource * Dsource * Constants.JY * freq_cgs * 4.0 * π)")
end

"""
    CalculateScaleFactor(sizex, sizey, pixelsx, pixelsy, SourceD, LengthUnit)

Compute the scale factor for the image, converting the per-pixel
intensity to a flux density.

# Arguments
- `sizex`, `sizey`: Image size, in `LengthUnit`.
- `pixelsx`, `pixelsy`: Image resolution.
- `SourceD`: Distance to the source, in cm.
- `LengthUnit`: Length unit, in cm (e.g. `model.L_unit`).

# Returns
- The scale factor.
"""
function CalculateScaleFactor(sizex, sizey, pixelsx, pixelsy, SourceD, LengthUnit)
    return (sizex * LengthUnit / pixelsx) * (sizey * LengthUnit / pixelsy) / (SourceD * SourceD) / Constants.JY
end


function raytrace_image_GPU!(
    d_traj, d_Image,
    i_offset, j_offset, block_size_x, block_size_y, # New offset parameters
    ro, θo, phi, bhspin, nx, ny, nmaxstep, 
    freq, fovx, fovy, Rout, Rstop, data, params
)
    local_i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    local_j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    i = local_i - 1 + i_offset
    j = local_j - 1 + j_offset

    if local_i <= block_size_x && local_j <= block_size_y && i < nx && j < ny
        
        calculate_image!(
            d_traj, d_Image, ro, θo, phi, bhspin, nx, ny, nmaxstep,
            i, j, local_i, local_j, freq, fovx, fovy, Rout, Rstop, params, data
        )
    end
    return nothing
end


function calculate_image!(
    traj, d_Image, 
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

    Kcon0 = Geodesics.init_Kcon(i_global, j_global, Xcam, nx, ny, fovx, fovy, bhspin, params)
    Kcon = Kcon0 * (freq * Constants.HPL / (Constants.ME * Constants.CL * Constants.CL))

    dl_unit::Float64 = params.L_unit * Constants.HPL / (Constants.ME * Constants.CL^2)
    Rh = 1.0 + sqrt(1.0 - bhspin * bhspin)

    X = Xcam
    K = Kcon
    Xhalf = SVector{4, Float64}(0.0, 0.0, 0.0, 0.0)
    Khalf = SVector{4, Float64}(0.0, 0.0, 0.0, 0.0)
    zero_vec = SVector{4, Float64}(0.0, 0.0, 0.0, 0.0)
    lconn = MArray{Tuple{4,4,4},Float64,3,64}(undef)

    step::Int64 = 1
    @inbounds traj[i_local, j_local, step] = OfTrajGRMHD(
        0.0, X, K, Xhalf, Khalf,
        zero_vec, zero_vec
    )
    while (Geodesics.stop_backward_integration(X, K, Rh, Rstop) == 0 && (step < nmaxstep))
        @inbounds begin
            dl = Geodesics.stepsize(X, K, params.cstartx, params.cstopx)
            scaled_dl = dl * dl_unit
            X, K, Xhalf, Khalf = Geodesics.push_photon(X, K, -dl, lconn, bhspin, params)
            step += 1
            traj[i_local, j_local, step] = OfTrajGRMHD(
                scaled_dl, X, K, Xhalf, Khalf,
                zero_vec, zero_vec
            )
        end
    end

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



end
