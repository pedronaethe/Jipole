"""
Top-level image-plane orchestration: batch intensity integration and
image/flux reporting.
"""
module Imaging

using Printf
using ..Constants
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

end
