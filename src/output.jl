"""
HDF5 output writers. `generate_output_file` is a generic writer for any
flat `Dict{String,Any}` of HDF5 paths/values; `generate_output_ipole`
writes the same structure `ipole` itself produces (see `ipole`'s
`io.c`/`write_header`/`dump` and `model/iharm/model.c`/`output_hdf5`),
so downstream tools built around ipole's convention work unchanged on
Jipole's output.
"""
module Output
using HDF5
using ..Constants
using ..Iharm


"""
    generate_output_file(output_file::String, data::Dict{String,Any}; format::String="generic")

Write an HDF5 file with the given `output_file` name, containing the
key-value pairs in the `data` dictionary. The optional `format` argument
can be set to `"ipole"` to write the file in the same format as `ipole`'s output, which is useful for compatibility with existing tools that expect `ipole` output.

As a note for users, new output formats can be easily added by implementing a new function in output.jl.
"""

function generate_output_file(output_file::String, data::Dict{String,Any}; format::String="generic")
    if format == "ipole"
        generate_output_ipole(output_file, data)
        return
    else if format != "generic"
        @error "Unsupported output format: $format. Supported formats are: 'generic', 'ipole'."
        return
    end

    # Create an HDF5 file
    h5file = h5open(output_file, "w")

    # Iterate over the data dictionary and write each key-value pair to the HDF5 file
    for (key, value) in data
        write(h5file, key, value)
    end

    # Close the HDF5 file
    close(h5file)
end

"""
    generate_output_ipole(output_file::String, data::Dict{String,Any})

Write an HDF5 file matching `ipole`'s own output convention exactly (same
group layout/field names as `ipole`'s `write_header`, `output_hdf5`, and
`dump`), so files can be read by the same tools/scripts written for
`ipole`'s output.

Jipole has no polarized radiative transfer yet, so the polarization-only
fields (`pol`, `Ftot`, `nuLnu`) are written as zero, matching what
`ipole` itself writes when run with `only_unpolarized 1`.

# Required keys in `data`
- `"image"`: unpolarized (Stokes I) image, `(nx, ny)`
- `"params"`: the `IharmParams` used to generate the image
- `"data"`: the `IharmData` snapshot used to generate the image (for the
  dump time `t` and the accretion diagnostics)
- `"ro"`, `"theta_o"`, `"phi"`: camera position (matches `rcam`/`thetacam`/`phicam`)
- `"fovx"`, `"fovy"`: field of view, radians (as computed by e.g. `DXsize/ro`)
- `"freq"`: observing frequency, cgs
- `"SourceD"`: source distance, cgs
- `"scale"`: the Jy-per-pixel-intensity scale factor (`calculate_scale_factor`)
- `"Xcamera"`: the 4-vector camera position

- `"trat_large"`: electron/ion temperature ratio at high magnetization
  (`Rhigh`) -- not stored in `IharmParams` itself (it's only a transient
  argument to `load_data`), so it must be passed here explicitly

# Optional keys
- `"tau"`: optical depth image, `(nx, ny)` -- defaults to zeros
- `"xoff"`, `"yoff"`, `"rotcam"` -- default to 0.0
"""
function generate_output_ipole(output_file::String, data::Dict{String,Any})
    image = data["image"]
    params = data["params"]
    snapshot = data["data"]
    ro = data["ro"]
    theta_o = data["theta_o"]
    phi = data["phi"]
    fovx = data["fovx"]
    fovy = data["fovy"]
    freq = data["freq"]
    SourceD = data["SourceD"]
    scale = data["scale"]
    Xcamera = data["Xcamera"]
    trat_large = data["trat_large"]

    tau = get(data, "tau", zeros(size(image)))
    xoff = get(data, "xoff", 0.0)
    yoff = get(data, "yoff", 0.0)
    rotcam = get(data, "rotcam", 0.0)

    nx, ny = size(image)
    NIMG = 5 # Stokes I,Q,U,V + Faraday depth, matching ipole's NIMG, even though we don't have polarization yet. Maybe soon?

    Mdot, MdotEdd, Ladv = Iharm.compute_accretion_diagnostics(params, snapshot)

    Ftot_unpol = sum(image) * scale
    nuLnu_unpol = 4π * Ftot_unpol * SourceD^2 * Constants.JY * freq
    Ftot = 0.0    
    nuLnu = 0.0

    fov_to_d = SourceD / params.L_unit / Constants.MUAS_PER_RAD
    DXsize = fovx * ro
    DYsize = fovy * ro
    dx = DXsize 
    dy = DYsize
    fovx_dsource = DXsize / fov_to_d
    fovy_dsource = DYsize / fov_to_d

    h5file = h5open(output_file, "w")

    write(h5file, "Ftot_unpol", Ftot_unpol)
    write(h5file, "Ftot", Ftot)
    write(h5file, "nuLnu", nuLnu)
    write(h5file, "nuLnu_unpol", nuLnu_unpol)
    write(h5file, "Mdot", Mdot)
    write(h5file, "MdotEdd", MdotEdd)
    write(h5file, "Ladv", Ladv)
    write(h5file, "unpol", image)
    write(h5file, "tau", tau)
    write(h5file, "pol", zeros(nx, ny, NIMG))

    write(h5file, "header/version", "Jipole-1.0")
    write(h5file, "header/githash", "unknown") # Jipole has no build-time githash yet
    write(h5file, "header/freqcgs", freq)
    write(h5file, "header/scale", scale)
    write(h5file, "header/dsource", SourceD)
    write(h5file, "header/evpa_0", "N")
    write(h5file, "header/t", snapshot.t)
    write(h5file, "header/sigma_cut", params.sigma_cut)
    write(h5file, "header/field_config", Int32(0)) # no B-field-reversal option in Jipole yet

    write(h5file, "header/camera/nx", Int32(nx))
    write(h5file, "header/camera/ny", Int32(ny))
    write(h5file, "header/camera/dx", dx)
    write(h5file, "header/camera/dy", dy)
    write(h5file, "header/camera/fovx_dsource", fovx_dsource)
    write(h5file, "header/camera/fovy_dsource", fovy_dsource)
    write(h5file, "header/camera/rcam", ro)
    write(h5file, "header/camera/thetacam", theta_o)
    write(h5file, "header/camera/phicam", phi)
    write(h5file, "header/camera/rotcam", rotcam)
    write(h5file, "header/camera/fovx", fovx)
    write(h5file, "header/camera/fovy", fovy)
    write(h5file, "header/camera/xoff", xoff)
    write(h5file, "header/camera/yoff", yoff)
    write(h5file, "header/camera/x", collect(Xcamera))

    write(h5file, "header/units/L_unit", params.L_unit)
    write(h5file, "header/units/M_unit", params.M_unit)
    write(h5file, "header/units/T_unit", params.T_unit)
    write(h5file, "header/units/Thetae_unit", params.Thetae_unit)

    write(h5file, "header/electrons/rlow", params.trat_small)
    write(h5file, "header/electrons/rhigh", trat_large)
    write(h5file, "header/electrons/beta_crit", params.beta_crit)
    write(h5file, "header/electrons/type", Int32(params.ELECTRONS))

    close(h5file)
end

end
