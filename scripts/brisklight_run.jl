# ---------------------------------------------------------------------------
# Brisk-light rendering. Included by main.jl when [brisklight].on = true;
# it reads main.jl's consts directly rather than taking arguments.
#
# The sequence is: trace every geodesic once, analyse the lensing bands' emission
# times at the requested p, then render each frame from only the dumps its bands
# actually need.
# ---------------------------------------------------------------------------

println("")
println("Brisk-light rendering: p = $brisk_p, n_bands = $brisk_nbands, cadence = $brisk_cadence M")
use_gpu && @warn "A GPU brisk-light kernel is not implemented yet; rendering on the CPU."

let
    dump_times = Jipole.Brisklight.dump_times_of(dump_files)
    println("Dump series: $(length(dump_files)) file(s), t = $(round(dump_times[1], digits=3)) .. $(round(dump_times[end], digits=3)) M")
    length(dump_times) >= 2 ||
        error("Brisk light needs at least 2 dumps to interpolate between; found $(length(dump_times)).")

    # Any dump serves as the header source: the grid is fixed across the series.
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

    # Traced once and reused for every frame. This is the run's dominant memory
    # cost: pixels_x * pixels_y * mean(nsteps) trajectory records.
    geo_time = @elapsed begin
        all_geodesics, nsteps, midplane_crossings, Xcamera, bounds =
            Jipole.Imaging.trace_all_geodesics(model, ro, th, phi, freq, pixels_x, pixels_y,
                fovx, fovy, maxnstep, Rh, xoff, yoff)
    end
    @printf("Traced %d geodesics in %.2f s (mean %.0f steps/ray)\n",
            pixels_x * pixels_y, geo_time, sum(nsteps) / length(nsteps))
    println("Crossing counts: ", [(c, count(==(c), midplane_crossings))
                                  for c in sort(unique(midplane_crossings))])

    # The emission weighting needs one snapshot to take j from. Use the dump
    # nearest the middle of the series: it only sets the relative weight of each
    # step, never a rendered value.
    ref_idx = max(1, div(length(dump_files) + 1, 2))
    ref_data = reader.load_data(dump_files[ref_idx], Rhigh, model)

    stats, cache, sample = Jipole.Brisklight.analyze_bands(
        midplane_crossings, all_geodesics, nsteps, pixels_x, pixels_y,
        brisk_nbands, model, brisk_p;
        sample_stride = brisk_stride, weighting = brisk_weighting,
        ref_data = ref_data, freq = freq)
    ref_data = nothing
    GC.gc()

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
    isempty(prefix) && (prefix = "brisk")

    series = Jipole.Brisklight.DumpSeries(dump_files; times = dump_times)
    result = Jipole.Brisklight.process_brisklight_images!(
        stats, cache, series, all_geodesics, nsteps, model,
        pixels_x, pixels_y, freq, Rhigh;
        cadence = brisk_cadence,
        t_start = brisk_t_start,
        n_frames = brisk_nframes,
        sample = sample,
        output_dir = outdir,
        output_prefix = prefix,
        output_format = output_format,
        meta = meta)

    if result.frames > 0
        @printf("\nWrote %d frame(s) to %s\n", result.frames, outdir)
        total_snaps = sum(r.n_snapshots for r in result.metrics)
        @printf("Snapshots per image: %.2f   total dump loads: %d   read %.2f GB\n",
                total_snaps / result.frames, series.n_loads, series.bytes_read / 2^30)
        @printf("Geodesics %.2f s   render %.2f s   I/O %.2f s   peak RSS %.2f GB\n",
                geo_time, sum(r.render_seconds for r in result.metrics),
                sum(r.load_seconds for r in result.metrics), Sys.maxrss() / 2^30)
    end
end
