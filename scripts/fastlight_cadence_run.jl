# ---------------------------------------------------------------------------
# Fast light on an observer-time cadence. Included by main.jl when
# [fastlight].cadence is set and slow light is off.
#
# Ordinary fast light renders one image per dump, at that dump's own time. With
# a cadence it instead renders on a time grid, interpolating the fluid between
# the two dumps bracketing each frame. That lets a fast-light run sit on exactly
# the same frame grid as a brisk or slow-light run, which is what makes the three
# directly comparable.
# ---------------------------------------------------------------------------

println("")
println("Fast-light rendering on a cadence of $fast_cadence M")
use_gpu && @warn "The cadence fast-light path is CPU-only for now; rendering on the CPU."

let
    isdir(dump_filepath) ||
        error("[fastlight].cadence requires [dump].dump_filepath to be a directory of dumps (got '$dump_filepath')")
    dump_times = Jipole.Brisklight.dump_times_of(dump_files)
    println("Dump series: $(length(dump_files)) file(s), t = $(round(dump_times[1], digits=3)) .. $(round(dump_times[end], digits=3)) M")

    reader = Jipole.Kharma.is_kharma_dump(dump_files[1]) ? Jipole.Kharma : Jipole.Iharm
    model = reader.read_header(dump_files[1], MBH; th_beg = th_beg, Rlow = Rlow, Rhigh = Rhigh,
        beta_crit = beta_crit, sigma_cut = sigma_cut, sigma_cut_high = sigma_cut_high,
        rmin_geo = rmin_geo, rmax_geo = rmax_geo,
        M_unit = M_unit)

    Rh = 1 + sqrt(1.0 - model.a^2)
    DXsize = SourceD / model.L_unit / Jipole.Constants.MUAS_PER_RAD * fov_size
    DYsize = DXsize
    fovx = DXsize / ro
    fovy = DYsize / ro

    geo_time = @elapsed begin
        all_geodesics, nsteps, midplane_crossings, Xcamera, bounds =
            Jipole.Imaging.trace_all_geodesics(model, ro, th, phi, freq, pixels_x, pixels_y,
                fovx, fovy, maxnstep, Rh, xoff, yoff)
    end
    @printf("Traced %d geodesics in %.2f s\n", pixels_x * pixels_y, geo_time)

    scale_factor = Jipole.Imaging.calculate_scale_factor(DXsize, DYsize, pixels_x, pixels_y,
                                                         SourceD, model.L_unit)
    meta = Dict{String,Any}(
        "params" => model, "ro" => ro, "theta_o" => th, "phi" => phi,
        "fovx" => fovx, "fovy" => fovy, "freq" => freq, "SourceD" => SourceD,
        "scale" => scale_factor, "Xcamera" => collect(Xcamera), "Rhigh" => Rhigh,
        "xoff" => xoff, "yoff" => yoff)

    outdir = dirname(output_filename)
    isempty(outdir) && (outdir = ".")
    prefix = first(splitext(basename(output_filename)))
    isempty(prefix) && (prefix = "fast")

    series = Jipole.Brisklight.DumpSeries(dump_files; times = dump_times)
    result = Jipole.Brisklight.process_fastlight_images!(
        series, all_geodesics, nsteps, model, pixels_x, pixels_y, freq, Rhigh;
        cadence = fast_cadence,
        t_start = fast_t_start,
        n_frames = fast_nframes,
        output_dir = outdir, output_prefix = prefix,
        output_format = output_format, meta = meta)

    result.frames > 0 && @printf("\nWrote %d frame(s) to %s; render %.2f s, peak RSS %.2f GB\n",
        result.frames, outdir, sum(r.render_seconds for r in result.metrics),
        Sys.maxrss() / 2^30)
end
