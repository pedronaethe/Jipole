"""
Gradient-based parameter recovery: fit observer inclination / spin
(`Analytic`/`ThinDisk`) or inclination / `Rhigh` (`Iharm`) to an observed
image, via finite differences or the autodiff sensitivities from
[`Autodiff`](@ref).
"""
module GradientDescent

using LinearAlgebra
using ImageFiltering
using ProgressMeter
using StaticArrays
using ..Constants
using ..GeoTypes
using ..Camera
using ..Geodesics
using ..Radiation
using ..Autodiff
using ..Imaging
using ..Iharm

export cost_func, GradientofCostFunction, FiniteDifferencesθ, FiniteDifferencesTrat,
    FiniteDifferences_a, armijo_line_search!, true_conjugate_gradient_optimization,
    true_conjugate_gradient_optimization_GRMHD

"""
    cost_func(ImageObs, ImageTest)

Normalized Mean Squared Error (NMSE) between the observed and test
images.

# Arguments
- `ImageObs`: Observed image.
- `ImageTest`: Test (model) image.

# Returns
- The NMSE.
"""
function cost_func(ImageObs, ImageTest)
    if length(ImageObs) != length(ImageTest)
        println("Length of ImageObs: $(length(ImageObs))")
        println("Length of ImageTest: $(length(ImageTest))")
        throw(ArgumentError("ImageObs and ImageTest must have the same length."))
    end

    numerator = sum((ImageTest .- ImageObs) .^ 2)
    denominator = sum(ImageObs .^ 2)

    nmse = numerator / denominator
    return nmse
end

"""
    GradientofCostFunction(ImageObs, ImageTest, dI_dθo, dI_da)

Gradient of [`cost_func`](@ref) with respect to `θo` and the second
sensitivity parameter (`a` or `Rhigh`).

# Arguments
- `ImageObs`: Observed image.
- `ImageTest`: Test (model) image.
- `dI_dθo`: Image sensitivity to `θo`.
- `dI_da`: Image sensitivity to `a`/`Rhigh`.

# Returns
- A tuple `(grad_θo, grad_a)`.
"""
function GradientofCostFunction(ImageObs, ImageTest, dI_dθo, dI_da)
    if size(ImageObs) != size(ImageTest)
        throw(ArgumentError("ImageObs and ImageTest must have the same shape"))
    end

    ΔI = ImageTest .- ImageObs
    denom = sum(ImageObs .^ 2)

    grad_θo = 2 * sum(ΔI .* dI_dθo) / denom
    grad_a = 2 * sum(ΔI .* dI_da) / denom

    return grad_θo, grad_a
end

"""
    FiniteDifferencesθ(ro, th, phi, DXsize, DYsize, pixels_x, pixels_y, SourceD, freq, maxnstep, h, bhspin, model, Rstop, data=nothing, xoff=0.0, yoff=0.0)

Finite-differences sensitivity of the image intensity to `th` (observer
inclination), using threaded per-pixel integration.

# Arguments
- `ro`, `th`, `phi`: Camera radial distance, inclination, and azimuth.
- `DXsize`, `DYsize`: Screen size, in `model.L_unit`.
- `pixels_x`, `pixels_y`: Image resolution.
- `SourceD`: Distance to the source, in cm.
- `freq`: Frequency, in cgs units.
- `maxnstep`: Maximum number of integration steps.
- `h`: Finite-difference step size, in degrees.
- `bhspin`: Dimensionless black hole spin parameter.
- `model`: Model parameters.
- `Rstop`: Backward-integration stopping radius.
- `data`: Model-specific auxiliary data.
- `xoff`, `yoff`: Image plane offsets.

# Returns
- A tuple `(dI_dθo, Imagec)`.
"""
function FiniteDifferencesθ(ro, th, phi, DXsize, DYsize, pixels_x, pixels_y, SourceD, freq, maxnstep, h, bhspin, model, Rstop, data=nothing, xoff=0.0, yoff=0.0)
    θh = th + h
    θl = th - h

    fovx = DXsize / ro
    fovy = DYsize / ro

    freq_unitless = freq * Constants.HPL / (Constants.ME * Constants.CL * Constants.CL)
    Rh = 1 + sqrt(1.0 - bhspin * bhspin)

    Xcamh = MVector{4,Float64}(Camera.camera_position(ro, θh, phi, bhspin, model))
    Xcaml = MVector{4,Float64}(Camera.camera_position(ro, θl, phi, bhspin, model))
    Xcamc = MVector{4,Float64}(Camera.camera_position(ro, th, phi, bhspin, model))

    scale_factor = Imaging.CalculateScaleFactor(DXsize, DYsize, pixels_x, pixels_y, SourceD, model.L_unit)
    println("scale_factor = $scale_factor")

    function trace_image(Xcamera, description)
        println("Calculating $description...")

        local_img = zeros(Float64, pixels_x, pixels_y)

        p = Progress(
            pixels_x * pixels_y;
            desc="Raytracing $description...",
            showspeed=true,
            barlen=30
        )

        Threads.@threads for i in 0:(pixels_x-1)
            for j in 0:(pixels_y-1)
                traj = Vector{OfTrajS}()
                sizehint!(traj, maxnstep)

                nstep, _ = Geodesics.get_pixel(traj, i, j, Xcamera, fovx, fovy, freq_unitless, pixels_x, pixels_y, bhspin, Rh, Rstop, model, xoff, yoff)

                resize!(traj, nstep)
                Radiation.integrate_emission!(traj, nstep, local_img, i + 1, j + 1, freq, bhspin, model, data)
            end
        end
        finish!(p)

        local_img *= freq^3
        return local_img
    end

    Imageh = trace_image(Xcamh, "High (+h)")
    Imagel = trace_image(Xcaml, "Low (-h)")

    dI_dθo = (Imageh - Imagel) / (2 * h)

    Imagec = trace_image(Xcamc, "Central")

    return dI_dθo, Imagec
end

"""
    FiniteDifferencesTrat(ro, th, phi, DXsize, DYsize, pixels_x, pixels_y, SourceD, freq, maxnstep, h_trat, bhspin, model, Rstop, trat_large, dump_filepath)

Finite-differences sensitivity of the image intensity to `trat_large`
(`Rhigh`), reloading the GRMHD data at each of `trat_large ± h_trat`.

# Arguments
- `ro`, `th`, `phi`: Camera radial distance, inclination, and azimuth.
- `DXsize`, `DYsize`: Screen size, in `model.L_unit`.
- `pixels_x`, `pixels_y`: Image resolution.
- `SourceD`: Distance to the source, in cm.
- `freq`: Frequency, in cgs units.
- `maxnstep`: Maximum number of integration steps.
- `h_trat`: Finite-difference step size for `trat_large`.
- `bhspin`: Dimensionless black hole spin parameter.
- `model`: Iharm model parameters.
- `Rstop`: Backward-integration stopping radius.
- `trat_large`: Central electron/ion temperature ratio at high
  magnetization.
- `dump_filepath`: Path to the GRMHD dump file.

# Returns
- A tuple `(dI_dRhigh, Imagec)`.
"""
function FiniteDifferencesTrat(ro, th, phi, DXsize, DYsize, pixels_x, pixels_y, SourceD, freq, maxnstep, h_trat, bhspin, model, Rstop, trat_large, dump_filepath)
    fovx = DXsize / ro
    fovy = DYsize / ro
    Xcam = MVector{4,Float64}(Camera.camera_position(ro, th, phi, bhspin, model))
    freq_unitless = freq * Constants.HPL / (Constants.ME * Constants.CL * Constants.CL)
    Rh = 1 + sqrt(1.0 - bhspin * bhspin)
    scale_factor = Imaging.CalculateScaleFactor(DXsize, DYsize, pixels_x, pixels_y, SourceD, model.L_unit)

    function trace_variant(target_trat, description)
        println("\n=== Processing $description ===")

        println("Loading simulation data for Trat = $target_trat...")
        local_data = Iharm.load_data(dump_filepath, target_trat, model)

        local_img = zeros(Float64, pixels_x, pixels_y)

        p = Progress(pixels_x * pixels_y; desc="Raytracing $description...", showspeed=true, barlen=30)

        Threads.@threads for i in 0:(pixels_x-1)
            for j in 0:(pixels_y-1)
                traj = Vector{OfTrajS}()
                sizehint!(traj, maxnstep)

                nstep, _ = Geodesics.get_pixel(traj, i, j, Xcam, fovx, fovy, freq_unitless, pixels_x, pixels_y, bhspin, Rh, Rstop, model, 0.0, 0.0)

                if length(traj) != nstep
                    resize!(traj, nstep)
                end

                Radiation.integrate_emission!(traj, nstep, local_img, i + 1, j + 1, freq, bhspin, model, local_data)
            end
        end
        finish!(p)

        local_data = nothing

        println("Forcing Garbage Collection for $description...")
        GC.gc()
        GC.gc()

        local_img .*= freq^3
        return local_img
    end

    Imageh = trace_variant(trat_large + h_trat, "High (+h)")
    Imagel = trace_variant(trat_large - h_trat, "Low (-h)")

    dI_dRhigh = (Imageh .- Imagel) ./ (2 * h_trat)

    Imageh = nothing
    Imagel = nothing
    println("Clearing High/Low buffers...")
    GC.gc()

    Imagec = trace_variant(trat_large, "Central")

    return dI_dRhigh, Imagec
end

"""
    FiniteDifferences_a(ro, th, phi, DXsize, DYsize, pixels_x, pixels_y, SourceD, freq, maxnstep, h, bhspin, model, Rstop)

Finite-differences sensitivity of the image intensity to `a` (black hole
spin), tracing full geodesics at each of `bhspin ± h`.

# Arguments
- `ro`, `th`, `phi`: Camera radial distance, inclination, and azimuth.
- `DXsize`, `DYsize`: Screen size, in `model.L_unit`.
- `pixels_x`, `pixels_y`: Image resolution.
- `SourceD`: Distance to the source, in cm.
- `freq`: Frequency, in cgs units.
- `maxnstep`: Maximum number of integration steps.
- `h`: Finite-difference step size.
- `bhspin`: Dimensionless black hole spin parameter.
- `model`: Model parameters.
- `Rstop`: Backward-integration stopping radius.

# Returns
- A tuple `(dI_da, Imagec)`.
"""
function FiniteDifferences_a(ro, th, phi, DXsize, DYsize, pixels_x, pixels_y, SourceD, freq, maxnstep, h, bhspin, model, Rstop)
    ah = bhspin + h
    al = bhspin - h

    fovx = DXsize / ro
    fovy = DYsize / ro

    Xcamh = MVector{4,Float64}(Camera.camera_position(ro, th, phi, ah, model))
    Xcaml = MVector{4,Float64}(Camera.camera_position(ro, th, phi, al, model))

    scale_factor = Imaging.CalculateScaleFactor(DXsize, DYsize, pixels_x, pixels_y, SourceD, model.L_unit)
    trajectoryh = Geodesics.CalculateGeodesics(Xcamh, fovx, fovy, freq, maxnstep, pixels_x, pixels_y, ah, Rstop, model)
    trajectoryl = Geodesics.CalculateGeodesics(Xcaml, fovx, fovy, freq, maxnstep, pixels_x, pixels_y, al, Rstop, model)

    Imageh = Imaging.IpoleGeoIntensityIntegration(trajectoryh, freq, pixels_x, pixels_y, ah, model)
    Imagel = Imaging.IpoleGeoIntensityIntegration(trajectoryl, freq, pixels_x, pixels_y, al, model)

    trajectoryh = nothing
    trajectoryl = nothing

    dI_da = (Imageh - Imagel) / (2 * h)

    Xcam = MVector{4,Float64}(Camera.camera_position(ro, th, phi, bhspin, model))
    trajectory = Geodesics.CalculateGeodesics(Xcam, fovx, fovy, freq, maxnstep, pixels_x, pixels_y, bhspin, Rstop, model)
    Imagec = Imaging.IpoleGeoIntensityIntegration(trajectory, freq, pixels_x, pixels_y, bhspin, model)
    trajectory = nothing

    return dI_da, Imagec
end

"""
    armijo_line_search!(cost_func, x, grad, direction, bounds, scales, args...; α=1e-4, β=0.5, initial_step=1.0, max_steps=10)

Backtracking line search along `direction`, with bounds handling.

# Arguments
- `cost_func`: Cost function, called as `cost_func(x, args...)`.
- `x`: Current point.
- `grad`: Gradient at `x`.
- `direction`: Search direction.
- `bounds`: Per-component `(lo, hi)` bounds.
- `scales`: Per-component scale factors (for logging only).
- `args`: Extra arguments forwarded to `cost_func`.

# Returns
- A tuple `(x_new, f_new, step_size, success)`.
"""
function armijo_line_search!(cost_func, x, grad, direction, bounds, scales,
    args...; α=1e-4, β=0.5, initial_step=1.0, max_steps=10)
    step_size = initial_step
    x_new = similar(x)

    f0 = cost_func(x, args...)
    df0 = dot(grad, direction)
    println("grad = $grad, direction = $direction")
    println("  Line search: f0=$f0, df0=$df0, initial_step=$initial_step")

    if direction[1] > 0
        println("  Direction for θo is increasing, testing higher θo values")
    else
        println("  Direction for θo is decreasing, testing lower θo values")
    end

    if df0 >= 0
        @warn "Not a descent direction, df0 = $df0"
    end

    best_x = copy(x)
    best_f = f0
    best_step = 0.0

    for i in 1:max_steps
        x_new .= x .+ step_size .* direction

        any_bounded = false
        for j in eachindex(x_new)
            old_val = x_new[j]
            x_new[j] = clamp(x_new[j], bounds[j][1], bounds[j][2])
            if x_new[j] != old_val
                any_bounded = true
            end
        end

        if any_bounded
            println("  Step $i: \e[31mHit bounds\e[0m")
            if x_new[2] == bounds[2][1] || x_new[2] == bounds[2][2]
                println(" Wanted to try Rhigh = $((x[2] .+ step_size .* direction[2]) * scales[2]), but hit bounds at $(x_new[2] * scales[2])")
                step_size = (x_new[2] - x[2]) / direction[2]
                println("")
            end
            if x_new[1] == bounds[1][1] || x_new[1] == bounds[1][2]
                println(" Wanted to try θo = $((x[1] .+ step_size .* direction[1]) * scales[1]), but hit bounds at $(x_new[1] * scales[1])")
                step_size = (x_new[1] - x[1]) / direction[1]
            end
        end

        step_norm = norm(x_new - x)
        if step_norm < 1e-20
            println(" \e[31mStep $i: Step norm too small ($step_norm), breaking\e[0m")
            println(" x_new = $x_new, x = $x, step_size = $step_size")
            break
        end

        f_new = cost_func(x_new, args...)
        println("  Step $i: step_size=$step_size, f_new=$f_new, improvement=$(f0-f_new), Rhigh tested = $(x_new[2] * scales[2]), θo tested = $(x_new[1] * scales[1])")

        if f_new < best_f
            best_x .= x_new
            best_f = f_new
            best_step = step_size
        end

        armijo_threshold = f0 + α * step_size * df0
        if f_new <= armijo_threshold
            println("  \e[32mArmijo condition satisfied!\e[0m")
            println(" New cost function value: $f_new")
            return x_new, f_new, step_size, true
        end

        println("Reducing step size by beta factor: $β, Armijo Condition: $armijo_threshold, $f0, step_size=$step_size, df0=$df0")
        step_size *= β

        if abs(x_new[1] - x[1]) < 1e-2 && abs(x_new[2] * scales[2] - x[2] * scales[2]) < 1e-2
            println("  \e[31mBoth parameter changes below 1e-2 threshold, returning best point found\e[0m")
            return best_x, best_f, best_step, false
        end
    end

    if best_f < f0
        println("\e[31m  No Armijo satisfaction, but found improvement: $(f0 - best_f)\e[0m")
        return best_x, best_f, best_step, false
    end

    println("\e[31mLine search failed to find any improvement\e[0m")
    return x, f0, best_step, false
end

"""
    true_conjugate_gradient_optimization(Iobs, ro, θoi, ai, freq, nx, ny, nmaxstep, fovx, fovy, model, Rstop, σ_pixels=0.0; kwargs...)

Recover the observer inclination `θo` and/or black hole spin `a` that
best reproduce the observed image `Iobs`, via conjugate-gradient descent
on [`cost_func`](@ref) using the autodiff sensitivities from
[`Autodiff.AutoDiffGeoTrajEulerMethod!`](@ref) (`Analytic`/`ThinDisk`
models).

# Arguments
- `Iobs`: Observed image intensities (the target to fit).
- `ro`: Observer radial coordinate.
- `θoi`: Initial guess for the inclination, in degrees.
- `ai`: Initial guess for the spin (`0` to `0.994`).
- `freq`: Observed frequency, e.g. 230 GHz.
- `nx`, `ny`: Image resolution.
- `nmaxstep`: Maximum number of geodesic integration steps.
- `fovx`, `fovy`: Field of view, in radians.
- `model`: Model parameters.
- `Rstop`: Stopping radius for the geodesic integration.
- `σ_pixels`: Standard deviation of the Gaussian filter applied to
  intensities and gradients.

# Keyword Arguments
- `cost_tol`, `param_tol`, `grad_tol`: Convergence tolerances.
- `max_iterations`: Maximum number of optimization iterations.
- `cg_restart_freq`: Conjugate-gradient restart frequency.
- `optimize_param`: `:both`, `:theta`, or `:spin`.

# Returns
- A tuple `(θos, as, costs, niter)` (or `(θo_final, a_final, costs, niter)`
  if the initial guess already satisfies the tolerance).
"""
function true_conjugate_gradient_optimization(Iobs, ro, θoi, ai, freq, nx, ny, nmaxstep,
    fovx, fovy, model, Rstop, σ_pixels=0.0;
    cost_tol=2e-11, param_tol=1e-8, grad_tol=1e-10,
    max_iterations=200, cg_restart_freq=20,
    optimize_param::Symbol=:both)
    num_threads = Threads.nthreads()
    thread_trajs = Vector{Vector{OfTraj}}(undef, num_threads)
    for tid in 1:num_threads
        thread_trajs[tid] = Vector{OfTraj}()
        sizehint!(thread_trajs[tid], nmaxstep)
    end

    if !(optimize_param in [:both, :theta, :spin])
        throw(ArgumentError("optimize_param must be :both, :theta, or :spin"))
    end

    θo_scale = 60.0
    a_scale = 0.6
    scales = [θo_scale, a_scale]

    x_scaled = [θoi / θo_scale, ai / a_scale]
    bounds_scaled = [(0.1 / θo_scale, 90.0 / θo_scale),
        (0.0 / a_scale, 0.994 / a_scale)]

    optimize_theta = optimize_param in [:both, :theta]
    optimize_spin = optimize_param in [:both, :spin]

    println("Optimization mode: $optimize_param")
    println("Optimizing θo: $optimize_theta, Optimizing a: $optimize_spin")

    dI_dθo = Matrix{Float64}(undef, nx, ny)
    dI_da = Matrix{Float64}(undef, nx, ny)
    I_calc = Matrix{Float64}(undef, nx, ny)

    function compute_cost_and_gradients(x_scaled_val, σ_pixels=0.0)
        θo_val = x_scaled_val[1] * θo_scale
        a_val = x_scaled_val[2] * a_scale
        println("Running AutoDiffGeoTrajEulerMethod with θo = $θo_val, a = $a_val and applying σ_pixels = $σ_pixels filter")
        Threads.@threads for i in 0:(nx-1)
            for j in 0:(ny-1)
                tid = Threads.threadid()
                dI_dθo_out = Ref{Float64}()
                intensity_out = Ref{Float64}()
                dI_da_out = Ref{Float64}()

                Autodiff.AutoDiffGeoTrajEulerMethod!(thread_trajs[tid], dI_dθo_out, intensity_out, dI_da_out,
                    ro, θo_val, 0.0, a_val, nx, ny, nmaxstep, i, j, freq, fovx, fovy, model, Rstop)

                I_calc[i+1, j+1] = intensity_out[]
                dI_da[i+1, j+1] = dI_da_out[]
                dI_dθo[i+1, j+1] = dI_dθo_out[]
            end
        end
        I_calc = imfilter(I_calc, Kernel.gaussian(σ_pixels))
        dI_dθo = imfilter(dI_dθo, Kernel.gaussian(σ_pixels))
        dI_da = imfilter(dI_da, Kernel.gaussian(σ_pixels))
        cost = cost_func(Iobs, I_calc)
        grad_θo, grad_a = GradientofCostFunction(Iobs, I_calc, dI_dθo, dI_da)

        grad_scaled = [grad_θo * θo_scale, grad_a * a_scale]

        if !optimize_theta
            grad_scaled[1] = 0.0
        end
        if !optimize_spin
            grad_scaled[2] = 0.0
        end

        return cost, grad_scaled
    end

    function constrained_armijo_line_search!(cost_func, x, grad, direction, bounds, scales, args...; kwargs...)
        constrained_direction = copy(direction)
        if !optimize_theta
            constrained_direction[1] = 0.0
        end
        if !optimize_spin
            constrained_direction[2] = 0.0
        end

        return armijo_line_search!(cost_func, x, grad, constrained_direction, bounds, scales, args...; kwargs...)
    end

    function check_convergence(cost, grad, cost_history, iteration)
        cost_converged = cost < cost_tol

        grad_norm = 0.0
        if optimize_theta
            grad_norm += grad[1]^2
        end
        if optimize_spin
            grad_norm += grad[2]^2
        end
        grad_norm = sqrt(grad_norm)
        grad_converged = false

        rel_improvement_converged = false
        if iteration > 10 && length(cost_history) > 10
            recent_improvement = (cost_history[end-9] - cost) / abs(cost_history[end-9])
            rel_improvement_converged = recent_improvement < param_tol
        end

        println("  Convergence check: cost=$cost, grad_norm=$grad_norm")
        if iteration > 10
            recent_improvement = length(cost_history) > 10 ? (cost_history[end-9] - cost) / abs(cost_history[end-9]) : Inf
            println("  Recent relative improvement: $recent_improvement")
        end
        println("  Cost converged: $cost_converged, Grad converged: $grad_converged, Stagnant: $rel_improvement_converged")

        return cost_converged || grad_converged || rel_improvement_converged
    end

    last_x_computed = nothing
    last_cost = nothing
    last_grad = nothing

    function cached_compute_cost_and_gradients(x_scaled_val, σ_pixels=0.0)
        if last_x_computed !== nothing && last_x_computed ≈ x_scaled_val
            println("Using cached computation for x = $x_scaled_val")
            return last_cost, last_grad
        end

        cost, grad = compute_cost_and_gradients(x_scaled_val, σ_pixels)
        last_x_computed = copy(x_scaled_val)
        last_cost = cost
        last_grad = copy(grad)
        return cost, grad
    end

    cost, grad = cached_compute_cost_and_gradients(x_scaled, σ_pixels)
    initial_cost = cost

    if check_convergence(cost, grad, [cost], 0)
        θo_final = x_scaled[1] * θo_scale
        a_final = x_scaled[2] * a_scale
        println("Initial solution already satisfies tolerance")
        return θo_final, a_final, [cost], 1
    end
    direction = -copy(grad)
    if !optimize_theta
        direction[1] = 0.0
    end
    if !optimize_spin
        direction[2] = 0.0
    end

    costs = Float64[]
    push!(costs, cost)
    θos = Float64[]
    push!(θos, x_scaled[1] * θo_scale)
    as = Float64[]
    push!(as, x_scaled[2] * a_scale)
    θo_phys = x_scaled[1] * θo_scale
    a_phys = x_scaled[2] * a_scale

    println("Initial cost: $cost, Initial θo: $θo_phys, Initial a: $a_phys")
    println("Initial gradient norm: $(norm(grad))")

    x_old = copy(x_scaled)
    step_size = 0.0
    aggressive_initial_step = 0.0
    for iteration in 1:max_iterations
        println("\n--- Iteration $iteration ---")

        if iteration == 1
            aggressive_initial_step = max(3.0, 0.3 / max(norm(grad), 1e-12))
        else
            if step_size < 0.05 / (a_scale * direction[2]) && optimize_spin
                step_size = 0.05 / (a_scale * direction[2])
                println("Step size for a is too small, resetting to $step_size")
            end

            if step_size < 3.0 / (θo_scale * direction[1]) && optimize_theta
                step_size = 3.0 / (θo_scale * direction[1])
                println("Step size for θo is too small, resetting to $step_size")
            end

            aggressive_initial_step = step_size
            aggressive_initial_step = max(3.0, 0.3 / max(norm(grad), 1e-12))
        end

        println("Trying aggressive initial step: $aggressive_initial_step, set at iteration $iteration")
        println("Cost before line search: $cost")
        cost_comparison = copy(cost)
        x_new, cost_new, step_size, success = constrained_armijo_line_search!(
            (x_val, args...) -> cached_compute_cost_and_gradients(x_val, σ_pixels)[1],
            x_scaled, grad, direction, bounds_scaled, scales,
            α=1e-5, β=0.5, initial_step=aggressive_initial_step, max_steps=15
        )

        println("Line search completed: new cost = $cost_new, step size = $step_size")
        println("Initial cost after line search: $cost")
        absolute_improvement = cost_comparison - cost_new
        relative_improvement = absolute_improvement / max(abs(cost_comparison), 1e-16)
        println("Cost improvement: $absolute_improvement (relative: $relative_improvement)")

        if absolute_improvement <= 1e-16 && iteration > 1
            println("No significant improvement in line search, trying steepest descent")
            direction .= -grad
            if !optimize_theta
                direction[1] = 0.0
            end
            if !optimize_spin
                direction[2] = 0.0
            end

            x_new, cost_new, step_size, success = constrained_armijo_line_search!(
                (x_val, args...) -> cached_compute_cost_and_gradients(x_val, σ_pixels)[1],
                x_scaled, grad, direction, bounds_scaled, scales,
                α=1e-5, β=0.5, initial_step=step_size * 10
            )

            absolute_improvement = cost - cost_new
            if absolute_improvement <= 1e-16
                @warn "No improvement possible, stopping optimization"
                break
            end
        end

        if optimize_theta
            x_scaled[1] = x_new[1]
        end
        if optimize_spin
            x_scaled[2] = x_new[2]
        end

        param_change = norm(x_new - x_old)
        x_old .= x_scaled
        cost = copy(cost_new)
        push!(costs, cost)

        θo_phys = x_scaled[1] * θo_scale
        a_phys = x_scaled[2] * a_scale
        push!(θos, θo_phys)
        push!(as, a_phys)
        println("Updated: cost = $cost, θo = $θo_phys, a = $a_phys, step = $step_size")
        println("Parameter change magnitude: $param_change")

        converged = check_convergence(cost, grad, costs, iteration)

        relative_param_change = param_change / max(norm(x_scaled), 1e-10)
        if relative_param_change < param_tol
            println("Relative parameter change too small ($relative_param_change), may have converged")
            converged = true
        end
        if converged
            println("Converged! Final θo = $θo_phys, Final a = $a_phys")
            return θos, as, costs, max_iterations
        end

        grad_old = copy(grad)
        _, grad = cached_compute_cost_and_gradients(x_scaled, σ_pixels)

        if iteration % cg_restart_freq == 0 || norm(grad) > 10 * norm(grad_old)
            direction .= -grad
            if !optimize_theta
                direction[1] = 0.0
            end
            if !optimize_spin
                direction[2] = 0.0
            end
            println("CG restart at iteration $iteration")
        else
            grad_diff = grad - grad_old
            beta_denom = dot(grad_old, grad_old)

            if beta_denom > 1e-16
                beta = max(0.0, dot(grad, grad_diff) / beta_denom)

                beta = min(beta, 2.0)

                direction .= -grad .+ beta .* direction

                if !optimize_theta
                    direction[1] = 0.0
                end
                if !optimize_spin
                    direction[2] = 0.0
                end

                active_grad_norm = 0.0
                active_dir_norm = 0.0
                active_dot = 0.0

                if optimize_theta
                    active_grad_norm += grad[1]^2
                    active_dir_norm += direction[1]^2
                    active_dot += direction[1] * grad[1]
                end
                if optimize_spin
                    active_grad_norm += grad[2]^2
                    active_dir_norm += direction[2]^2
                    active_dot += direction[2] * grad[2]
                end

                active_grad_norm = sqrt(active_grad_norm)
                active_dir_norm = sqrt(active_dir_norm)

                if active_dot >= -1e-10 * active_dir_norm * active_grad_norm
                    direction .= -grad
                    if !optimize_theta
                        direction[1] = 0.0
                    end
                    if !optimize_spin
                        direction[2] = 0.0
                    end
                    println("Reset to steepest descent (not descent direction)")
                end

                println("CG update: beta = $beta")
            else
                direction .= -grad
                if !optimize_theta
                    direction[1] = 0.0
                end
                if !optimize_spin
                    direction[2] = 0.0
                end
                println("Beta = 0, Reset to steepest descent (small denominator)")
            end
        end
    end

    @warn "Maximum iterations reached without convergence"
    return θos, as, costs, max_iterations
end

"""
    true_conjugate_gradient_optimization_GRMHD(Iobs, ro, θoi, Rhighi, freq, nx, ny, nmaxstep, fovx, fovy, phi, DXsize, DYsize, SourceD, model, Rstop, dump_filepath; kwargs...)

Recover the observer inclination `θo` and/or `Rhigh` that best reproduce
the observed image `Iobs`, via conjugate-gradient descent on
[`cost_func`](@ref), using either the autodiff sensitivities from
[`Autodiff.AutoDiffGeoTrajEulerMethod_GRMHD!`](@ref) or finite differences
(`Iharm` model).

# Arguments
- `Iobs`: Observed image intensities (the target to fit).
- `ro`: Observer radial coordinate.
- `θoi`: Initial guess for the inclination, in degrees.
- `Rhighi`: Initial guess for `Rhigh`.
- `freq`: Observed frequency, e.g. 230 GHz.
- `nx`, `ny`: Image resolution.
- `nmaxstep`: Maximum number of geodesic integration steps.
- `fovx`, `fovy`: Field of view, in radians.
- `phi`: Observer azimuth, in degrees.
- `DXsize`, `DYsize`: Screen size, in `model.L_unit` (only used by the
  finite-differences sensitivity mode).
- `SourceD`: Distance to the source, in cm (only used by the
  finite-differences sensitivity mode).
- `model`: Iharm model parameters.
- `Rstop`: Stopping radius for the geodesic integration.
- `dump_filepath`: Path to the GRMHD dump file (reloaded whenever
  `Rhigh` changes).
- `σ_pixels`: Standard deviation of the Gaussian filter applied to
  intensities and gradients.

# Keyword Arguments
- `cost_tol`, `param_tol`, `grad_tol`: Convergence tolerances.
- `max_iterations`: Maximum number of optimization iterations.
- `cg_restart_freq`: Conjugate-gradient restart frequency.
- `optimize_param`: `:both`, `:theta`, or `:Rhigh`.
- `sensemode`: `"AD"` (autodiff) or `"FD"` (finite differences).
- `xoff`, `yoff`: Image plane offsets (plain-render mode only).

# Returns
- A tuple `(θos, Rhighs, costs, niter)`.
"""
function true_conjugate_gradient_optimization_GRMHD(Iobs, ro, θoi, Rhighi, freq, nx, ny, nmaxstep,
    fovx, fovy, phi, DXsize, DYsize, SourceD, model, Rstop, dump_filepath;
    cost_tol=2e-11, param_tol=1e-8, grad_tol=1e-10,
    max_iterations=200, cg_restart_freq=20,
    optimize_param::Symbol=:both, sensemode="AD", xoff=0.0, yoff=0.0)
    num_threads = Threads.nthreads()
    simulation_data = Iharm.load_data(dump_filepath, Rhighi, model)

    thread_trajs = Vector{Vector{OfTraj}}(undef, num_threads + 1)
    for tid in 1:(num_threads+1)
        thread_trajs[tid] = Vector{OfTraj}()
        sizehint!(thread_trajs[tid], nmaxstep)
    end

    if !(optimize_param in [:both, :theta, :Rhigh])
        throw(ArgumentError("optimize_param must be :both, :theta, or :Rhigh"))
    end

    if !(sensemode in ["AD", "FD"])
        throw(ArgumentError("sensemode arg must be either 'AD' or 'FD'"))
    end

    success::Bool = false
    Rh = 1 + sqrt(1.0 - model.a * model.a)
    freq_unitless = freq * Constants.HPL / (Constants.ME * Constants.CL * Constants.CL)

    θo_scale = 100.0
    Rhigh_scale = 200.0
    scales = [θo_scale, Rhigh_scale]

    x_scaled = [θoi / θo_scale, Rhighi / Rhigh_scale]
    bounds_scaled = [(0.1 / θo_scale, 175.0 / θo_scale),
        (0.0 / Rhigh_scale, 100.0 / Rhigh_scale)]

    optimize_theta = optimize_param in [:both, :theta]
    optimize_Rhigh = optimize_param in [:both, :Rhigh]

    println("Optimization mode: $optimize_param")
    println("Optimizing θo: $optimize_theta, Optimizing Rhigh: $optimize_Rhigh")

    dI_dθo = Matrix{Float64}(undef, nx, ny)
    dI_dRhigh = Matrix{Float64}(undef, nx, ny)
    I_calc = Matrix{Float64}(undef, nx, ny)

    function compute_cost_and_gradients(x_scaled_val, compute_gradients, σ_pixels=0.0, simulation_data=nothing, sensemode="AD")
        θo_val = x_scaled_val[1] * θo_scale
        Rhigh_val = x_scaled_val[2] * Rhigh_scale
        if optimize_Rhigh
            simulation_data = Iharm.load_data(dump_filepath, Rhigh_val, model)
        end
        if compute_gradients
            println("\n Running AutoDiffGeoTrajEulerMethod with θo = \e[32m$θo_val\e[0m, Rhigh = \e[32m$Rhigh_val\e[0m and applying σ_pixels = \e[32m$σ_pixels\e[0m filter")
            if sensemode == "AD"
                Threads.@threads for i in 0:(nx-1)
                    for j in 0:(ny-1)
                        tid = Threads.threadid()
                        dI_dθo_out = Ref{Float64}()
                        intensity_out = Ref{Float64}()
                        dI_dRhigh_out = Ref{Float64}()

                        Autodiff.AutoDiffGeoTrajEulerMethod_GRMHD!(thread_trajs[tid], dI_dθo_out, intensity_out, dI_dRhigh_out,
                            ro, θo_val, phi, model.a, nx, ny, nmaxstep, i, j, freq, fovx, fovy, model, Rstop, simulation_data)
                        I_calc[i+1, j+1] = intensity_out[]
                        dI_dRhigh[i+1, j+1] = dI_dRhigh_out[]
                        dI_dθo[i+1, j+1] = dI_dθo_out[]
                    end
                end
            elseif sensemode == "FD"
                println("Using Finite Differences to compute gradients")
                dI_dθo, I_calc = FiniteDifferencesθ(ro, θo_val, phi, DXsize, DYsize, nx, ny, SourceD, freq, nmaxstep, 1e-4, model.a, model, Rstop, simulation_data)
                if optimize_Rhigh
                    error("Finite Differences for Rhigh not implemented yet")
                end
            else
                throw(ArgumentError("sensemode must be either 'AD' or 'FD'"))
            end
            I_calc = imfilter(I_calc, Kernel.gaussian(σ_pixels))
            dI_dθo = imfilter(dI_dθo, Kernel.gaussian(σ_pixels))
            dI_dRhigh = imfilter(dI_dRhigh, Kernel.gaussian(σ_pixels))
            cost = cost_func(Iobs, I_calc)
            grad_θo, grad_Rhigh = GradientofCostFunction(Iobs, I_calc, dI_dθo, dI_dRhigh)

            grad_scaled = [grad_θo * θo_scale, grad_Rhigh * Rhigh_scale]

            if !optimize_theta
                grad_scaled[1] = 0.0
            end
            if !optimize_Rhigh
                grad_scaled[2] = 0.0
            end

            return cost, grad_scaled
        else
            println("\n Computing image with \e[32mθo = $θo_val\e[0m, \e[32mRhigh = $Rhigh_val\e[0m and applying σ_pixels = \e[32m$σ_pixels\e[0m filter")
            Xcamera = MVector{4,Float64}(Camera.camera_position(ro, θo_val, phi, model.a, model))
            scale_factor = Imaging.CalculateScaleFactor(DXsize, DYsize, nx, ny, SourceD, model.L_unit)
            Threads.@threads for i in 0:(nx-1)
                tid = Threads.threadid()
                for j in 0:(ny-1)
                    empty!(thread_trajs[tid])
                    nstep, _ = Geodesics.get_pixel(thread_trajs[tid], i, j, Xcamera, fovx, fovy, freq_unitless, nx, ny, model.a, Rh, Rstop, model, xoff, yoff)
                    Radiation.integrate_emission!(thread_trajs[tid], nstep, I_calc, i + 1, j + 1, freq, model.a, model, simulation_data)
                    empty!(thread_trajs[tid])
                end
            end
        end
        I_calc *= freq^3
        I_calc = imfilter(I_calc, Kernel.gaussian(σ_pixels))
        cost = cost_func(Iobs, I_calc)
        return cost, zeros(2)
    end

    function constrained_armijo_line_search!(cost_func, x, grad, direction, bounds, scales, args...; kwargs...)
        constrained_direction = copy(direction)
        if !optimize_theta
            constrained_direction[1] = 0.0
        end
        if !optimize_Rhigh
            constrained_direction[2] = 0.0
        end

        return armijo_line_search!(cost_func, x, grad, constrained_direction, bounds, scales, args...; kwargs...)
    end

    function check_convergence(cost, grad, cost_history, iteration)
        cost_converged = cost < cost_tol

        grad_norm = 0.0
        if optimize_theta
            grad_norm += grad[1]^2
        end
        if optimize_Rhigh
            grad_norm += grad[2]^2
        end
        grad_norm = sqrt(grad_norm)
        grad_converged = false

        rel_improvement_converged = false
        if iteration > 10 && length(cost_history) > 10
            recent_improvement = (cost_history[end-9] - cost) / abs(cost_history[end-9])
            rel_improvement_converged = recent_improvement < param_tol
        end

        println("  Convergence check: cost=$cost, grad_norm=$grad_norm")
        if iteration > 10
            recent_improvement = length(cost_history) > 10 ? (cost_history[end-9] - cost) / abs(cost_history[end-9]) : Inf
            println("  Recent relative improvement: $recent_improvement")
        end
        println("  Cost converged: $cost_converged, Grad converged: $grad_converged, Stagnant: $rel_improvement_converged")

        return cost_converged || grad_converged || rel_improvement_converged
    end

    function cached_compute_cost_and_gradients(x_scaled_val, compute_gradients, σ_pixels=0.0, simulation_data=nothing, sensemode="AD")
        c, g = compute_cost_and_gradients(x_scaled_val, compute_gradients, σ_pixels, simulation_data, sensemode)
        return c, g
    end

    cost, grad = cached_compute_cost_and_gradients(x_scaled, true, σ_pixels, simulation_data, sensemode)
    initial_cost = cost

    if check_convergence(cost, grad, [cost], 0)
        θo_final = x_scaled[1] * θo_scale
        Rhigh_final = x_scaled[2] * Rhigh_scale
        println("Initial solution already satisfies tolerance")
        return θo_final, Rhigh_final, [cost], 1
    end
    direction = -copy(grad)
    if !optimize_theta
        direction[1] = 0.0
    end
    if !optimize_Rhigh
        direction[2] = 0.0
    end

    costs = Float64[]
    push!(costs, cost)
    θos = Float64[]
    push!(θos, x_scaled[1] * θo_scale)
    Rhighs = Float64[]
    push!(Rhighs, x_scaled[2] * Rhigh_scale)
    θo_phys = x_scaled[1] * θo_scale
    Rhigh_phys = x_scaled[2] * Rhigh_scale

    println("Initial cost: $cost, Initial θo: $θo_phys, Initial Rhigh: $Rhigh_phys")
    println("Initial gradient norm: $(norm(grad))")

    x_old = copy(x_scaled)
    step_size = 0.0
    aggressive_initial_step = 0.0
    direction_old = copy(direction)
    grad_old = copy(grad)
    for iteration in 1:max_iterations
        println("\n--- Iteration $iteration ---")

        if iteration == 1
            max_dθo_phys = 20.0
            max_dRhigh_phys = 10.0

            max_dθo_scaled = max_dθo_phys / θo_scale
            max_dRhigh_scaled = max_dRhigh_phys / Rhigh_scale

            step_limit_θo = (optimize_theta && abs(direction[1]) > 1e-16) ? (max_dθo_scaled / abs(direction[1])) : Inf
            step_limit_Rhigh = (optimize_Rhigh && abs(direction[2]) > 1e-16) ? (max_dRhigh_scaled / abs(direction[2])) : Inf

            aggressive_initial_step = max(step_limit_θo, step_limit_Rhigh)
        else
            if optimize_Rhigh && abs(direction[2]) > 1e-16
                min_step_Rhigh = 5.0 / (Rhigh_scale * abs(direction[2]))
                if step_size < min_step_Rhigh
                    step_size = min_step_Rhigh
                    println("Step size for Rhigh is too small, resetting to $step_size")
                end
            end

            if optimize_theta && abs(direction[1]) > 1e-16
                min_step_theta = 5.0 / (θo_scale * abs(direction[1]))
                if step_size < min_step_theta
                    step_size = min_step_theta
                    println("Step size for θo is too small, resetting to $step_size")
                end
            end
            if success
                dir_deriv_old = dot(grad_old, direction_old)
                dir_deriv_new = dot(grad, direction)

                if abs(dir_deriv_new) > 1e-40
                    correction_factor = dir_deriv_old / dir_deriv_new

                    correction_factor = clamp(correction_factor, 0.1, 5.0)

                    aggressive_initial_step = step_size * correction_factor
                    println("Line search successful. CG scaled step size by $correction_factor -> new guess: $aggressive_initial_step")
                else
                    aggressive_initial_step = step_size * 1.5
                end
            else
                println("Line search failed to find a better point, resetting aggressive initial step")
                max_dθo_phys = 10.0
                max_dRhigh_phys = 10.0

                max_dθo_scaled = max_dθo_phys / θo_scale
                max_dRhigh_scaled = max_dRhigh_phys / Rhigh_scale

                step_limit_θo = (optimize_theta && abs(direction[1]) > 1e-16) ? (max_dθo_scaled / abs(direction[1])) : Inf
                step_limit_Rhigh = (optimize_Rhigh && abs(direction[2]) > 1e-16) ? (max_dRhigh_scaled / abs(direction[2])) : Inf

                aggressive_initial_step = max(step_limit_θo, step_limit_Rhigh)
            end
        end

        cost_comparison = copy(cost)
        x_new, cost_new, step_size, success = constrained_armijo_line_search!(
            (x_val, args...) -> cached_compute_cost_and_gradients(x_val, false, σ_pixels, simulation_data, sensemode)[1],
            x_scaled, grad, direction, bounds_scaled, scales,
            α=1e-5, β=0.5, initial_step=aggressive_initial_step, max_steps=15
        )

        absolute_improvement = cost_comparison - cost_new
        relative_improvement = absolute_improvement / max(abs(cost_comparison), 1e-16)
        println("\e[34mCost improvement: $absolute_improvement (relative: $relative_improvement)\e[0m")

        if absolute_improvement <= 1e-16 && iteration > 1
            println("Stagnation detected — probing directions")

            x_trial = copy(x_scaled)
            best_cost = cost
            best_x = copy(x_scaled)

            θ_step = 5.0 / θo_scale
            R_step = 5.0 / Rhigh_scale

            max_rounds = 10
            shrink = 0.8

            found_escape = false

            for round in 1:max_rounds
                println("Probe round $round with θ_step=$(θ_step*θo_scale), R_step=$(R_step*Rhigh_scale)")

                directions = Vector{Vector{Float64}}()

                if optimize_theta
                    push!(directions, [θ_step, 0.0])
                    push!(directions, [-θ_step, 0.0])
                end

                if optimize_Rhigh
                    push!(directions, [0.0, R_step])
                    push!(directions, [0.0, -R_step])
                end

                for d in directions
                    x_trial .= x_scaled .+ d

                    x_trial[1] = clamp(x_trial[1], bounds_scaled[1]...)
                    x_trial[2] = clamp(x_trial[2], bounds_scaled[2]...)

                    trial_cost, _ = cached_compute_cost_and_gradients(
                        x_trial, false, σ_pixels, simulation_data, sensemode
                    )

                    println("  Probe direction $d → cost = $trial_cost")

                    if trial_cost < best_cost
                        best_cost = trial_cost
                        best_x .= x_trial
                        found_escape = true
                    end
                end

                if found_escape
                    break
                end

                θ_step *= shrink
                R_step *= shrink
            end

            if found_escape
                println("Escape found! Restarting CG")
                x_scaled .= best_x
                cost = best_cost
                converged = check_convergence(cost, grad, costs, iteration)

                if converged
                    println("\e[32mConverged after escape! Final θo = $(x_scaled[1] * θo_scale), Final Rhigh = $(x_scaled[2] * Rhigh_scale)\e[0m")
                    println("\e[32mFinal cost: $cost\e[0m")
                    θo_phys = x_scaled[1] * θo_scale
                    Rhigh_phys = x_scaled[2] * Rhigh_scale
                    push!(costs, cost)

                    push!(θos, θo_phys)
                    push!(Rhighs, Rhigh_phys)
                    return θos, Rhighs, costs, max_iterations
                end

                grad_old .= grad
                direction_old .= direction

                _, grad = cached_compute_cost_and_gradients(
                    x_scaled, true, σ_pixels, simulation_data, sensemode
                )

                direction .= -grad
                continue
            else
                println("No escape found after $max_rounds rounds — stopping")
                break
            end
        end

        if optimize_theta
            x_scaled[1] = x_new[1]
        end
        if optimize_Rhigh
            x_scaled[2] = x_new[2]
        end

        param_change = norm(x_new - x_old)
        x_old .= x_scaled
        cost = copy(cost_new)
        push!(costs, cost)

        θo_phys = x_scaled[1] * θo_scale
        Rhigh_phys = x_scaled[2] * Rhigh_scale
        push!(θos, θo_phys)
        push!(Rhighs, Rhigh_phys)
        println("Updated: cost = $cost, θo = $θo_phys, Rhigh = $Rhigh_phys, step = $step_size")
        println("Parameter change magnitude: $param_change")

        converged = check_convergence(cost, grad, costs, iteration)

        relative_param_change = param_change / max(norm(x_scaled), 1e-10)
        if relative_param_change < param_tol
            println("Relative parameter change too small ($relative_param_change), may have converged")
            println("However, we won't stop")

            converged = false
        end
        if converged
            println("Converged! Final θo = $θo_phys, Final Rhigh = $Rhigh_phys")
            return θos, Rhighs, costs, iteration
        end

        grad_old = copy(grad)
        direction_old = copy(direction)
        _, grad = cached_compute_cost_and_gradients(x_scaled, true, σ_pixels, simulation_data, sensemode)

        if iteration % cg_restart_freq == 0 || norm(grad) > 10 * norm(grad_old)
            direction .= -grad
            if !optimize_theta
                direction[1] = 0.0
            end
            if !optimize_Rhigh
                direction[2] = 0.0
            end
            println("CG restart at iteration $iteration")
        else
            grad_diff = grad - grad_old
            beta_denom = dot(grad_old, grad_old)

            if beta_denom > 1e-16
                beta = max(0.0, dot(grad, grad_diff) / beta_denom)

                beta = min(beta, 2.0)

                direction .= -grad .+ beta .* direction

                if !optimize_theta
                    direction[1] = 0.0
                end
                if !optimize_Rhigh
                    direction[2] = 0.0
                end

                active_grad_norm = 0.0
                active_dir_norm = 0.0
                active_dot = 0.0

                if optimize_theta
                    active_grad_norm += grad[1]^2
                    active_dir_norm += direction[1]^2
                    active_dot += direction[1] * grad[1]
                end
                if optimize_Rhigh
                    active_grad_norm += grad[2]^2
                    active_dir_norm += direction[2]^2
                    active_dot += direction[2] * grad[2]
                end

                active_grad_norm = sqrt(active_grad_norm)
                active_dir_norm = sqrt(active_dir_norm)

                if active_dot >= -1e-10 * active_dir_norm * active_grad_norm
                    direction .= -grad
                    if !optimize_theta
                        direction[1] = 0.0
                    end
                    if !optimize_Rhigh
                        direction[2] = 0.0
                    end
                    println("Reset to steepest descent (not descent direction)")
                end

                println("CG update: beta = $beta")
            else
                direction .= -grad
                if !optimize_theta
                    direction[1] = 0.0
                end
                if !optimize_Rhigh
                    direction[2] = 0.0
                end
                println("Reset to steepest descent (small denominator)")
            end
        end
    end

    @warn "Maximum iterations reached without convergence"
    return θos, Rhighs, costs, max_iterations
end

end
