"""
Polarized imaging: the polarized counterpart of `imaging.jl`.

`raytrace_image_pol` produces the unpolarized image and the polarized image
(Stokes I, Q, U, V and Faraday depth) in one pass over the pixels, like ipole
run without `-unpol`. It does not contain a pixel loop of its own: it hands the
existing `Imaging.raytrace_image` a `Polarization.PolSink`, which makes each
pixel run the polarized integrator after the Stokes-I one, on the same geodesic.

`polarization_summary`/`output_stokes_parameters_pol` reproduce the
image-integrated numbers ipole prints in `print_image_stats` (`main.c`).
"""
module ImagingPol

using ..Constants
using ..Imaging
using ..Polarization

export raytrace_image_pol, polarization_summary, output_stokes_parameters_pol

"""
    raytrace_image_pol(model, simulation_data, ro, th, phi, freq, pixels_x, pixels_y,
                       fovx, fovy, maxnstep, Rh, xoff, yoff; qu_conv=0)

Ray-trace the unpolarized and the polarized image of `model`.

The arguments are those of `Imaging.raytrace_image`, which does the actual
work; see `Polarization.PolSink` for how the polarized integrator is attached
to its pixel loop.

# Arguments
- as `Imaging.raytrace_image`.
- `qu_conv`: Q/U sign convention. `0` (default, as in ipole) measures the EVPA
  East of North; `1` North of West.

# Returns
- `(Image, pol)`: the unpolarized intensity image, of size
  `(pixels_x, pixels_y)`, exactly as `Imaging.raytrace_image` returns it, and
  the polarized image, of size `(NIMG, pixels_x, pixels_y)`, holding Stokes
  I, Q, U, V (CGS specific intensity) and the Faraday depth for each pixel.
"""
function raytrace_image_pol(model, simulation_data, ro, th, phi, freq, pixels_x, pixels_y,
                            fovx, fovy, maxnstep, Rh, xoff, yoff; qu_conv::Int=0)
    # Same element type as the image allocated inside `Imaging.raytrace_image`.
    T = promote_type(typeof(ro), typeof(th), typeof(phi), typeof(model.a))

    pol = zeros(T, Polarization.NIMG, pixels_x, pixels_y)
    sink = Polarization.PolSink(simulation_data, pol, qu_conv)

    Image = Imaging.raytrace_image(model, sink, ro, th, phi, freq, pixels_x, pixels_y,
                                   fovx, fovy, maxnstep, Rh, xoff, yoff)
    return Image, pol
end

"""
    polarization_summary(pol, scale_factor)

Image-integrated polarization of a polarized image, as printed by ipole's
`print_image_stats`.

# Arguments
- `pol`: Polarized image of size `(NIMG, nx, ny)`.
- `scale_factor`: Conversion from CGS intensity to Jy per pixel
  (`Imaging.calculate_scale_factor`).

# Returns
- A named tuple with the total fluxes `Ftot`, `Qtot`, `Utot`, `Vtot` [Jy], the
  net linear and circular polarization fractions `LP`, `CP` [%], and the net
  electric-vector position angle `EVPA` [deg], `0.5 atan(U, Q)`.
"""
function polarization_summary(pol, scale_factor)
    Ftot = sum(@view pol[1, :, :]) * scale_factor
    Qtot = sum(@view pol[2, :, :]) * scale_factor
    Utot = sum(@view pol[3, :, :]) * scale_factor
    Vtot = sum(@view pol[4, :, :]) * scale_factor
    LP = 100.0 * sqrt(Qtot * Qtot + Utot * Utot) / Ftot
    CP = 100.0 * Vtot / Ftot
    EVPA = rad2deg(0.5 * atan(Utot, Qtot))
    return (; Ftot, Qtot, Utot, Vtot, LP, CP, EVPA)
end

"""
    output_stokes_parameters_pol(pol, freq_cgs, scale_factor, Dsource)

Print the image-integrated polarization of `pol` to standard output, after the
unpolarized summary of `Imaging.output_stokes_parameters`.

# Arguments
- `pol`: Polarized image of size `(NIMG, nx, ny)`.
- `freq_cgs`: Observing frequency [Hz].
- `scale_factor`: Conversion from CGS intensity to Jy per pixel.
- `Dsource`: Distance to the source [cm].
"""
function output_stokes_parameters_pol(pol, freq_cgs, scale_factor, Dsource)
    s = polarization_summary(pol, scale_factor)
    println("Polarized transfer: Ftot = $(s.Ftot) Jy")
    println("nuLnu (polarized) = $(s.Ftot * Dsource * Dsource * Constants.JY * freq_cgs * 4.0 * π)")
    println("I,Q,U,V [Jy]: $(s.Ftot) $(s.Qtot) $(s.Utot) $(s.Vtot)")
    println("LP,CP [%]: $(s.LP) $(s.CP)")
    println("EVPA [deg]: $(s.EVPA)")
end

end
