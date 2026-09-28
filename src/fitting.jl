module Fitting

using Printf
using ProgressMeter
using StaticArrays

"""
    dump_flux(filepath, M_unit, res; save_template=nothing)

Total flux (Jy) of one dump imaged at `res`×`res` pixels with mass unit `M_unit`, using the
run's camera, frequency and plasma parameters. Its output is silenced: the fit calls it
hundreds of times. With `save_template`, the image is also written to that path, numbered
with the dump index.
"""
function dump_flux(filepath, M_unit, res; save_template=nothing)
    return redirect_stdio(stdout=devnull, stderr=devnull) do
        reader = Jipole.Kharma.is_kharma_dump(filepath) ? Jipole.Kharma : Jipole.Iharm
        model = reader.read_header(filepath, MBH; th_beg=th_beg, Rlow=Rlow, Rhigh=Rhigh, beta_crit=beta_crit, sigma_cut=sigma_cut, sigma_cut_high=sigma_cut_high, M_unit=M_unit)
        simulation_data = [reader.load_data(filepath, Rhigh, model)]
        Rh = 1 + sqrt(1.0 - model.a^2)
        DXsize = SourceD / model.L_unit / Jipole.Constants.MUAS_PER_RAD * fov_size
        fov = DXsize / ro
        image = Jipole.Imaging.raytrace_image(model, simulation_data, ro, th, phi, freq, res, res, fov, fov, maxnstep, Rh, xoff, yoff)
        scale = Jipole.Imaging.calculate_scale_factor(DXsize, DXsize, res, res, SourceD, model.L_unit)

        if save_template !== nothing
            Xcamera = MVector{4,Float64}(Jipole.Camera.camera_position(ro, th, phi, model.a, model))
            output_data = Dict{String,Any}(
                "image" => image, "img_time" => 0.0, "params" => model, "data" => simulation_data[1],
                "ro" => ro, "theta_o" => th, "phi" => phi, "fovx" => fov, "fovy" => fov, "freq" => freq,
                "SourceD" => SourceD, "scale" => scale, "Xcamera" => Xcamera, "Rhigh" => Rhigh,
            )
            fit_output = Jipole.Utils.dump_output_filename(save_template, Jipole.Utils.extract_dump_index(basename(filepath)))
            mkpath(dirname(fit_output))
            Jipole.Output.generate_output_file(fit_output, output_data; format=output_format)
        end

        sum(image) * scale
    end
end

"""
    mean_flux(files, M_unit, res; save_template=nothing, desc="Flux fit")

Average of [`dump_flux`](@ref) over `files`, freeing each dump before loading the next (see
the render loop below for why the empty threaded loop is needed). Shows a progress bar
labelled `desc`.
"""
function mean_flux(files, M_unit, res; save_template=nothing, desc="Flux fit")
    total = 0.0
    p = Progress(length(files); desc="$desc: ", showspeed=true, barlen=30)
    for (n, f) in enumerate(files)
        total += dump_flux(f, M_unit, res; save_template=save_template)
        Threads.@threads :greedy for _ in 1:Threads.nthreads()
        end
        GC.gc()
        next!(p; showvalues=[(:dumps, "$n/$(length(files))"), (:mean_flux_so_far, "$(round(total / n, sigdigits=4)) Jy")])
    end
    return total / length(files)
end

"""
    fit_M_unit(files, M_guess, target; res, tol, maxiter=10, save_template=nothing)

Find the M_unit for which the mean flux over `files` equals `target` (Jy), to a relative
tolerance `tol`. Flux grows roughly as a power of M_unit, so this is a secant search on
log(flux) against log(M_unit); each iteration is one pass over `files`. With
`save_template`, every pass writes its images there, so the files left at the end are the
ones at the returned M_unit.
"""
function fit_M_unit(files, M_guess, target; res, tol, maxiter=10, save_template=nothing)
    logM = [log(M_guess)]
    #Calculate the first mean flux
    logF = [log(mean_flux(files, M_guess, res; save_template=save_template, desc=@sprintf("Flux fit pass 0 (M_unit = %.4e g)", M_guess)))]
    for iteration in 0:maxiter
       
        @printf("Flux fit %d: M_unit = %.6e g gives mean flux %.6g Jy (target %.6g Jy)\n", iteration, exp(logM[end]), exp(logF[end]), target)
        #calculate the ratio of the current flux to the target flux
        #if it's within the tolerance, return the current M_unit, or if it reaches maxiter.
        abs(exp(logF[end]) / target - 1) < tol && return exp(logM[end])
        iteration == maxiter && break
        # Local power-law slope d log(flux) / d log(M_unit). The first step assumes 2, typical for
        # optically thin synchrotron; flux must grow with M_unit, hence the floor.
        slope = length(logM) == 1 ? 2.0 : max((logF[end] - logF[end-1]) / (logM[end] - logM[end-1]), 0.1)
        # At most a factor 1000 in M_unit per step
        step = clamp((log(target) - logF[end]) / slope, -log(1000.0), log(1000.0))
        push!(logM, logM[end] + step)
        push!(logF, log(mean_flux(files, exp(logM[end]), res; save_template=save_template,  desc=@sprintf("Flux fit pass %d (M_unit = %.4e g)", length(logM) - 1, exp(logM[end])))))
    end
    @warn "Flux fit did not reach $(100tol)% of $target Jy in $maxiter iterations; using M_unit = $(exp(logM[end]))"
    return exp(logM[end])
end

end