using Jipole
using StaticArrays
using ProgressMeter
using TOML
using Printf
if length(ARGS) != 1
    error("Usage: julia --project=. --threads=12 main.jl path/to/config.toml")
end

config_filepath = ARGS[1]
config = TOML.parsefile(config_filepath)

println("Using configuration: $config_filepath")

# This Sections is basically going to get all the parameters from the parse file. The functions used for that are in src/utils.jl
# If any parameter is missing, it will use the default value and print a warning.

const MBH = Jipole.Utils.get_config(config, "physical", "MBH", 6.2e9)
const slow_light = Jipole.Utils.get_config(config, "physical", "slow_light", false)


m_unit_def = Jipole.Utils.get_config(config, "physical", "m_unit", "MAD")

if m_unit_def == "MAD"
    const M_unit_guess = Jipole.Constants.M_UNIT_MAD
elseif m_unit_def == "SANE"
    const M_unit_guess = Jipole.Constants.M_UNIT_SANE
elseif m_unit_def isa Number
    const M_unit_guess = float(m_unit_def)
else
    error("m_unit par must be either 'MAD', 'SANE', or a numeric value, got: $m_unit_def")
end


# Dump parameters. Slow light requires dump_filepath to be a directory
# (validated below) and derives its dump range from the
# same t_init/t_final-filtered listing that fast light uses.
# expanduser is used to expand the ~ to the user's home directory
const dump_filepath = expanduser(Jipole.Utils.get_config(config, "dump", "dump_filepath", ""))

# t_init/t_final only mean anything when dump_filepath is a directory of dumps
if isdir(dump_filepath)
    const t_init = Jipole.Utils.get_config(config, "dump", "t_init", -Inf)
    const t_final = Jipole.Utils.get_config(config, "dump", "t_final", Inf)
else
    const t_init = -Inf
    const t_final = Inf
end
# Plasma parameters
const Rhigh = Jipole.Utils.get_config(config, "plasma", "Rhigh", 20.0)
const Rlow = Jipole.Utils.get_config(config, "plasma", "Rlow", 1.0)
const beta_crit = Jipole.Utils.get_config(config, "plasma", "beta_crit", 1.0)
const th_beg = Jipole.Utils.get_config(config, "plasma", "th_beg", 1.74e-2)
const sigma_cut = Jipole.Utils.get_config(config, "plasma", "sigma_cut", 1.0)
const sigma_cut_high = Jipole.Utils.get_config(config, "plasma", "sigma_cut_high", -1.0)

# Camera parameters
const ro = Jipole.Utils.get_config(config, "camera", "ro", 1000.0)
const th = Jipole.Utils.get_config(config, "camera", "theta_o", 60.0)
const phi = Jipole.Utils.get_config(config, "camera", "phi", 0.0)

# Image parameters
const pixels_x = Jipole.Utils.get_config(config, "image", "pixels_x", 128)
const pixels_y = Jipole.Utils.get_config(config, "image", "pixels_y", 128)
const fov_size = Jipole.Utils.get_config(config, "image", "fov_size", 160.0)
const xoff = Jipole.Utils.get_config(config, "image", "xoff", 0.0)
const yoff = Jipole.Utils.get_config(config, "image", "yoff", 0.0)


# Observing parameters
const freq = Jipole.Utils.get_config(config, "observing", "freq", 230e9)
const source_distance_pc = Jipole.Utils.get_config(config, "observing", "source_distance_pc", 16.9e6)

# Raytracing parameters
const maxnstep = Jipole.Utils.get_config(config, "raytracing", "maxnstep", 25000)
const mode = Jipole.Utils.get_config(config, "raytracing", "mode", "cpu")


#check how to do gradients
const do_gradients = Jipole.Utils.get_config(config, "gradient", "on", false)
const gradient_wrt = Tuple(Symbol.(Jipole.Utils.get_config(config, "gradient", "wrt", String[])))
if do_gradients
    slow_light && error("Gradients are not supported in slow light mode.")
    mode == "gpu" && error("Gradients are only implemented for mode = \"cpu\".")
    isempty(gradient_wrt) && error("[gradient].wrt must list at least one parameter, e.g. wrt = [\"M_unit\"]")
    unknown = setdiff(gradient_wrt, Jipole.Iharm.GRADIENT_PARAM_NAMES)
    isempty(unknown) || error("Unknown gradient parameter(s) $(unknown); choose from $(Jipole.Iharm.GRADIENT_PARAM_NAMES)")
end

# Polarized transfer (Stokes Q, U, V and Faraday depth). The [polarization] section is optional:
# without it the run is unpolarized, exactly as before.
const do_polarization = get(get(config, "polarization", Dict{String,Any}()), "on", false)
if do_polarization
    slow_light && error("Polarization is not supported in slow light mode.")
end

# CUDA.jl takes several seconds to load, so only load it when this run uses the GPU. Loading it
# also loads Jipole's CUDA extension (ext/JipoleCUDAExt.jl), which holds all the GPU code.
const use_gpu = slow_light ? Jipole.Utils.get_config(config, "slowlight", "engine", "cpu") == "gpu" : mode == "gpu"

if do_gradients && use_gpu
    println("Using GPU for gradient computations in this version has not been tested. This will land soon, for now, use CPU mode.")
end

if use_gpu
    using CUDA
end
# Output parameters
const output_filename = Jipole.Utils.get_config(config, "output", "filename", "jipole_output.h5")
const output_format = Jipole.Utils.get_config(config, "output", "format", "ipole")

const SourceD = source_distance_pc * Jipole.Constants.PC
const freq_unitless = freq * Jipole.Constants.HPL / (Jipole.Constants.ME * Jipole.Constants.CL^2)

const dump_files = Jipole.Utils.resolve_dump_files(dump_filepath, t_init, t_final)

println("Found $(length(dump_files)) dump file(s).")


const flux_fit = Jipole.Utils.get_config(config, "fit", "flux_fit", false)


if flux_fit
    const flux_value = Jipole.Utils.get_config(config, "fit", "flux_value", 0.5)
    const fitting_res = Jipole.Utils.get_config(config, "fit", "fitting_res", 80)
    const fit_amount = Jipole.Utils.get_config(config, "fit", "fit_amount", 100)
    const fitting_tol = Jipole.Utils.get_config(config, "fit", "fitting_tol", 0.01)
    const fit_save = Jipole.Utils.get_config(config, "fit", "fit_save", false)

    # fit_amount dumps, equally spaced over the selected ones; all of them if there are fewer
    const fit_files = dump_files[unique(round.(Int, range(1, length(dump_files); length=min(fit_amount, length(dump_files)))))]
    # Fit images go to a "fit" folder next to the regular output, e.g. results/run/fit/
    const fit_template = fit_save ? joinpath(dirname(output_filename), "fit", basename(output_filename)) : nothing
    println("Fitting M_unit to a mean flux of $flux_value Jy over $(length(fit_files)) dump(s) at $(fitting_res)x$(fitting_res) pixels...")
     const fit_settings = (; MBH, th_beg, Rlow, Rhigh, beta_crit, sigma_cut, sigma_cut_high, SourceD, fov_size,
                            ro, th, phi, freq, maxnstep, xoff, yoff, output_format)
    const M_unit = Jipole.Fitting.fit_M_unit(fit_files, M_unit_guess, flux_value, fit_settings;
                                             res=fitting_res, tol=fitting_tol, save_template=fit_template)
    @printf("Using fitted M_unit = %.6e g\n", M_unit)
else
    const M_unit = M_unit_guess
end

# Fast light branching
if !slow_light
    const batched = length(dump_files) > 1

    
    Image = zeros(Float64, pixels_x, pixels_y)
    if mode == "gpu"
        # nmaxstep for the first GPU attempt. render_image_gpu! doubles it for any tile with a
        # truncated geodesic, and the value it ends on carries over to the next dump.
        gpu_maxnstep = 4000
    elseif mode != "cpu"
        error("Invalid mode: $mode. Must be 'cpu' or 'gpu'.")
    end


    #Loop through every file in the directory chosen.
    for current_dump_filepath in dump_files
        GC.gc()
        local model, simulation_data, Rh, DXsize, DYsize, fovx, fovy, Xcamera, p, scale_factor
        local pol, grads_pol
        println("")
        println("Processing dump: $current_dump_filepath")

        # Read the header and load the data for this dump file.
        reader = Jipole.Kharma.is_kharma_dump(current_dump_filepath) ? Jipole.Kharma : Jipole.Iharm
        model = reader.read_header(current_dump_filepath, MBH; th_beg=th_beg, Rlow=Rlow, Rhigh=Rhigh, beta_crit=beta_crit, sigma_cut=sigma_cut, sigma_cut_high=sigma_cut_high, M_unit=M_unit)

        #This will read the primitives and the variables derived from them.
        simulation_data = [reader.load_data(current_dump_filepath, Rhigh, model)]

        Rh = 1 + sqrt(1.0 - model.a^2)
        DXsize = SourceD / model.L_unit / Jipole.Constants.MUAS_PER_RAD * fov_size
        DYsize = SourceD / model.L_unit / Jipole.Constants.MUAS_PER_RAD * fov_size
        fovx = DXsize / ro
        fovy = DYsize / ro

        # Define position of the camera in internal coordinates.
        Xcamera = MVector{4,Float64}(Jipole.Camera.camera_position(ro, th, phi, model.a, model))



        #Again, we have to separate CPU and GPU here, as they have different memory management strategies and different execution paths.
        if mode == "cpu"
            if do_gradients && do_polarization
                ctx = Jipole.Iharm.grmhd_context(model, simulation_data[1])
                grad_image, pol, grads, grads_pol = Jipole.IharmPol.calculate_gradients_pol(ctx, freq, pixels_x, pixels_y, fov_size, maxnstep, xoff, yoff;
                    MBH=MBH, Rhigh=Rhigh, Rlow=Rlow, beta_crit=beta_crit, th_beg=th_beg, sigma_cut=sigma_cut, sigma_cut_high=sigma_cut_high,
                    M_unit=M_unit, ro=ro, th=th, phi=phi, sourceD=SourceD,
                    wrt=gradient_wrt);
                copyto!(Image, grad_image)
            elseif do_gradients
                ctx = Jipole.Iharm.grmhd_context(model, simulation_data[1])
                grad_image, grads = Jipole.Iharm.calculate_gradients(ctx, freq, pixels_x, pixels_y, fov_size, maxnstep, xoff, yoff;
                    MBH=MBH, Rhigh=Rhigh, Rlow=Rlow, beta_crit=beta_crit, th_beg=th_beg, sigma_cut=sigma_cut, sigma_cut_high=sigma_cut_high,
                    M_unit=M_unit, ro=ro, th=th, phi=phi, sourceD=SourceD,
                    wrt=gradient_wrt);
                copyto!(Image, grad_image)
            elseif do_polarization
                unpol_image, pol = Jipole.ImagingPol.raytrace_image_pol(model, simulation_data, ro, th, phi, freq, pixels_x, pixels_y, fovx, fovy, maxnstep, Rh, xoff, yoff)
                copyto!(Image, unpol_image)
            else
                copyto!(Image, Jipole.Imaging.raytrace_image(model, simulation_data, ro, th, phi, freq, pixels_x, pixels_y, fovx, fovy, maxnstep, Rh, xoff, yoff))
            end
        else
            println("Copying simulation data grid to the GPU...")
            gpu_sim_data = (Jipole.Utils_GPU.copy_iharm_to_gpu(simulation_data[1]),)

            println("Processing image in tiles...")
            gpu_time = @elapsed begin
                global gpu_maxnstep = Jipole.Imaging.render_image_gpu!(Image, model, gpu_sim_data, ro, th, phi, freq, fovx, fovy, pixels_x, pixels_y;
                    nmaxstep=gpu_maxnstep, nmaxstep_ceiling=50000, block_size=128)
            end
            @printf("GPU raytracing took %.3f s\n", gpu_time)
            println("Raytracing complete!")
        end

        scale_factor = Jipole.Imaging.calculate_scale_factor(DXsize, DYsize, pixels_x, pixels_y, SourceD, model.L_unit)
        println("")

        # output_stokes_parameters only prints its summary to stdout, so capture
        # that output and draw the whole thing (header + summary) as one box
        original_stdout = stdout
        (stokes_output_rd, stokes_output_wr) = redirect_stdout()
        try
            Jipole.Imaging.output_stokes_parameters(Image, freq, scale_factor, pixels_x, pixels_y, SourceD)
            do_polarization && Jipole.ImagingPol.output_stokes_parameters_pol(pol, freq, scale_factor, SourceD)
        finally
            redirect_stdout(original_stdout)
        end
        close(stokes_output_wr)
        stokes_output_lines = split(read(stokes_output_rd, String), '\n'; keepempty=false)


        # Make a cool box with the summary of the dump file and the stokes parameters output. (just so it looks cool)
        let raw_box_lines = ["Summary of file $current_dump_filepath:"; stokes_output_lines], pad = 2
            term_width = try
                displaysize(stdout)[2]
            catch
                80
            end
            max_content_width = max(10, term_width - 2 * pad - 2)
            box_lines = [length(line) > max_content_width ? line[1:max_content_width-1] * "…" : line for line in raw_box_lines]
            inner_width = maximum(length, box_lines) + 2 * pad
            printstyled("┌" * "─"^inner_width * "┐\n"; color=:green)
            for line in box_lines
                printstyled("│"; color=:green)
                printstyled(" "^pad * line * " "^(inner_width - pad - length(line)); color=:blue)
                printstyled("│\n"; color=:green)
            end
            printstyled("└" * "─"^inner_width * "┘\n"; color=:green)
        end


        # Aseemble a dictionary that will be passed to generate_output_file function.
        output_data = Dict{String,Any}(
            "image" => Image,
            "img_time" => 0.0,
            "params" => model,
            "data" => simulation_data[1],
            "ro" => ro,
            "theta_o" => th,
            "phi" => phi,
            "fovx" => fovx,
            "fovy" => fovy,
            "freq" => freq,
            "SourceD" => SourceD,
            "scale" => scale_factor,
            "Xcamera" => Xcamera,
            "Rhigh" => Rhigh,
        )

        #Add gradients to the output data if they were calculated.
        do_gradients && (output_data["grads"] = grads)

        #Add the polarized image, and its gradients, if they were calculated.
        do_polarization && (output_data["pol"] = pol)
        do_polarization && do_gradients && (output_data["grads_pol"] = grads_pol)


        # Determine the output filename based on whether we are processing a batch of dumps or a single dump.
        dump_index = batched ? Jipole.Utils.extract_dump_index(basename(current_dump_filepath)) : nothing

        # Generate the current file image output filename based on the snapshots index.
        this_output_filename = Jipole.Utils.dump_output_filename(output_filename, dump_index)

        # Finally generate the output HDF5 file, creating the destination directory first if needed.
        output_dir = dirname(this_output_filename)
        isempty(output_dir) || mkpath(output_dir)
        Jipole.Output.generate_output_file(this_output_filename, output_data; format=output_format)

        println("Wrote $this_output_filename in ipole's HDF5 format")
    end
else
    #slow-light rending
    isdir(dump_filepath) || error("slow_light requires [dump].dump_filepath to be a directory of dumps (got a single file: '$dump_filepath')")

    const all_dumps_path = Jipole.Utils.dump_path_template(dump_files[1])
    const dump_start = Jipole.Utils.extract_dump_index(basename(dump_files[1]))
    const dump_max = Jipole.Utils.extract_dump_index(basename(dump_files[end]))
    const image_cadence = Jipole.Utils.get_config(config, "slowlight", "image_cadence", 10.0)
    const slowlight_engine = use_gpu ? :GPU : :CPU

    params_slowlight = Jipole.Slowlight.OfSlowLight(dump_start, dump_max, image_cadence, 0.0, 0.0, 0.0, "")
    params_slowlight.current_dumps_path = Jipole.Slowlight.update_dump_path(params_slowlight, all_dumps_path)

    model = Jipole.Iharm.read_header(params_slowlight.current_dumps_path, MBH; th_beg=th_beg, Rlow=Rlow, Rhigh=Rhigh, beta_crit=beta_crit, sigma_cut=sigma_cut, sigma_cut_high=sigma_cut_high, M_unit=M_unit, slow_light=true)

    advance_dump_path! = () -> (params_slowlight.current_dumps_path = Jipole.Slowlight.update_dump_path(params_slowlight, all_dumps_path))

    simulation_data = Vector{Jipole.Iharm.IharmData{Float64,Array{Float64,3},Float64,Array{Float64,3}}}(undef, 3)
    simulation_data[1] = Jipole.Iharm.load_data(params_slowlight.current_dumps_path, Rhigh, model; advance_path! = advance_dump_path!)
    simulation_data[2] = Jipole.Iharm.load_data(params_slowlight.current_dumps_path, Rhigh, model; advance_path! = advance_dump_path!)
    simulation_data[3] = Jipole.Iharm.load_data(params_slowlight.current_dumps_path, Rhigh, model; advance_path! = advance_dump_path!)

    params_slowlight.tA = simulation_data[1].t
    params_slowlight.tB = simulation_data[2].t
    params_slowlight.tf = Jipole.Slowlight.get_specific_dump_time(params_slowlight.dump_max, all_dumps_path)

    Rh = 1 + sqrt(1.0 - model.a^2)
    DXsize = SourceD / model.L_unit / Jipole.Constants.MUAS_PER_RAD * fov_size
    DYsize = SourceD / model.L_unit / Jipole.Constants.MUAS_PER_RAD * fov_size
    fovx = DXsize / ro
    fovy = DYsize / ro
    Xcamera = MVector{4,Float64}(Jipole.Camera.camera_position(ro, th, phi, model.a, model))
    scale_factor = Jipole.Imaging.calculate_scale_factor(DXsize, DYsize, pixels_x, pixels_y, SourceD, model.L_unit)

    println("Tracing geodesics (dump-independent; traced once for the whole slow-light run)...")

    dummy_svec = zero(SVector{4,Float64})
    dummy_traj = Jipole.GeoTypes.OfTrajGeneric{Float64}(0.0, dummy_svec, dummy_svec, dummy_svec, dummy_svec)

    row_trajs = [Vector{Jipole.GeoTypes.OfTrajGeneric{Float64}}(undef, maxnstep) for _ in 1:pixels_x]
    for i in 1:pixels_x
        for k in 1:maxnstep
            row_trajs[i][k] = dummy_traj
        end
    end

    all_geodesics = Matrix{Vector{Jipole.GeoTypes.OfTrajGeneric{Float64}}}(undef, pixels_x, pixels_y)
    nsteps = zeros(Int, pixels_x, pixels_y)

    row_t0 = zeros(Float64, pixels_x)
    row_tgeoi = fill(-1e100, pixels_x)
    row_tgeof = zeros(Float64, pixels_x)

    p = Progress(pixels_x * pixels_y; desc="Tracing geodesics...", showspeed=true, barlen=30)
    progress_lock = ReentrantLock()
    #Calculate the geodesic in slowlight has to be done first.
    Threads.@threads :greedy for i in 0:(pixels_x - 1)
        my_traj = row_trajs[i + 1]

        for j in 0:(pixels_y - 1)
            nstep, _ = Jipole.Geodesics.get_pixel(my_traj, i, j, Xcamera, fovx, fovy, freq_unitless, pixels_x, pixels_y, model.a, Rh, model.rmax_geo, model, xoff, yoff)

            nsteps[i + 1, j + 1] = nstep
            all_geodesics[i + 1, j + 1] = my_traj[1:nstep]

            final_step_time = my_traj[nstep].X[1]
            if final_step_time < row_t0[i + 1]
                row_t0[i + 1] = final_step_time
            end

            pixel_tgeoi = 1.0
            pixel_tgeof = 1.0
            for k in 1:nstep
                X = my_traj[k].X
                K = my_traj[k].Kcon

                log_r = X[2]
                t_coord = X[1]
                k_r = K[2]

                if pixel_tgeoi > 0.0 && log_r < log(100.0)
                    pixel_tgeoi = t_coord
                end
                if pixel_tgeof > 0.0 && log_r > log(100.0) && k_r < 0.0
                    pixel_tgeof = t_coord
                end
            end

            if pixel_tgeoi < 0.0 && pixel_tgeoi > row_tgeoi[i + 1]
                row_tgeoi[i + 1] = pixel_tgeoi
            end
            if pixel_tgeof < 0.0 && pixel_tgeof < row_tgeof[i + 1]
                row_tgeof[i + 1] = pixel_tgeof
            elseif pixel_tgeof > 0.0 && final_step_time < row_tgeof[i + 1]
                row_tgeof[i + 1] = final_step_time
            end

            lock(progress_lock) do
                ProgressMeter.next!(p; showvalues=[(:pixel, "($i, $j)"), (:total_done, "$(i * pixels_y + j)/$(pixels_x * pixels_y)")])
            end
        end
    end
    finish!(p)

    t0 = minimum(row_t0)
    tgeof = minimum(row_tgeof)
    tgeoi = maximum(row_tgeoi)

    println("Calculated t0 (absolute longest time): $t0")
    println("Calculated tgeof (oldest active time): $tgeof")
    println("Calculated tgeoi (newest active time): $tgeoi")

    row_trajs = nothing
    GC.gc()

    #Finally generate the image
    println("Starting slow-light image processing loop...")
    Jipole.Slowlight.process_slowlight_images!(
        params_slowlight, simulation_data, all_geodesics, nsteps,
        model, t0, tgeof, tgeoi, pixels_x, pixels_y, freq, Rhigh, all_dumps_path,
        Xcamera, ro, th, phi, fovx, fovy, SourceD, scale_factor;
        engine=slowlight_engine
    )
    println("Done!")
end
