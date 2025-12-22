using ImageFiltering



function cost_func(ImageObs, ImageTest)
    """
    Mean squared error cost function comparing each pixel's observed intensity with the calculated intensity.
    """
    
    if length(ImageObs) != length(ImageTest)
        println("Length of ImageObs: $(length(ImageObs))")
        println("Length of ImageTest: $(length(ImageTest))")
        throw(ArgumentError("ImageObs and ImageTest must have the same length."))
    end
    cost = 0.0
    for i in eachindex(ImageObs, ImageTest)
        cost += (ImageTest[i] - ImageObs[i])^2
    end
    return cost
end

function GradientofCostFunction(ImageObs, ImageTest, dI_dθo, dI_da)
    """
    Calculate the gradient of the cost function with respect to the observed image.
    The gradient is computed as the difference between the observed and calculated intensities.
    """
    
    if length(ImageObs) != length(ImageTest)
        println("Length of ImageObs: $(length(ImageObs))")
        println("Length of ImageTest: $(length(ImageTest))")
        throw(ArgumentError("ImageObs and ImageTest must have the same length."))
    end

    grad_θo = 2.0 * sum((ImageTest .- ImageObs) .* (dI_dθo))
    grad_a = 2.0 * sum((ImageTest .- ImageObs) .* (dI_da))
    return grad_θo, grad_a
end


function FiniteDifferencesθ(ro, th, phi, DXsize, DYsize, pixels_x, pixels_y, SourceD, freq, maxnstep,h, bhspin, Rout, Rstop, data = nothing)
    """
    Finite differences method to calculate the intensity at each pixel in the image.
    This function calculates the geodesics and the intensity at each pixel using finite differences.
    """
    θh = th + h
    θl = th - h

    # Calculate the fov at each dim x and y 
    fovx = DXsize / ro
    fovy = DYsize / ro

    # Calculate the camera position in native coordinates
    Xcamh = MVec4(camera_position(ro, θh, phi, bhspin, Rout))
    Xcaml = MVec4(camera_position(ro, θl, phi, bhspin, Rout))

    # Scales the intensity of each pixel by the real size of each pixel
    scale_factor = CalculateScaleFactor(DXsize, DYsize, pixels_x, pixels_y, SourceD, L_unit)
    # integrate_emission_flag setted to false signals that the output of the function will be the trajectory and not the Image
    trajectoryh = CalculateGeodesics(Xcamh, fovx, fovy, freq, maxnstep, pixels_x, pixels_y, bhspin, Rout, Rstop);
    trajectoryl = CalculateGeodesics(Xcaml, fovx, fovy, freq, maxnstep, pixels_x, pixels_y, bhspin, Rout, Rstop);

    # Integrate the emission along the geodesics
    Imageh = IpoleGeoIntensityIntegration(trajectoryh, freq, pixels_x, pixels_y, scale_factor, bhspin, data)
    Imagel = IpoleGeoIntensityIntegration(trajectoryl, freq, pixels_x, pixels_y, scale_factor, bhspin, data)

    dI_dθo = (Imageh - Imagel) / (2 * h)  # Finite difference approximation

    return dI_dθo, Imageh, Imagel
end

function FiniteDifferences_a(ro, th, phi, DXsize, DYsize, pixels_x, pixels_y, SourceD, freq, maxnstep,h, bhspin, Rout, Rstop)
    """
    Finite differences method to calculate the intensity at each pixel in the image.
    This function calculates the geodesics and the intensity at each pixel using finite differences.
    """
    ah = bhspin + h
    al = bhspin - h

    # Calculate the fov at each dim x and y 
    fovx = DXsize / ro
    fovy = DYsize / ro

    # Calculate the camera position in native coordinates
    Xcamh = MVec4(camera_position(ro, th, phi, ah, Rout))
    Xcaml = MVec4(camera_position(ro, th, phi, al, Rout))

    # Scales the intensity of each pixel by the real size of each pixel
    scale_factor = CalculateScaleFactor(DXsize, DYsize, pixels_x, pixels_y, SourceD, L_unit)
    # integrate_emission_flag setted to false signals that the output of the function will be the trajectory and not the Image
    trajectoryh = CalculateGeodesics(Xcamh, fovx, fovy, freq, maxnstep, pixels_x, pixels_y, ah, Rout, Rstop);
    # for step in 1:length(trajectoryh[1,1])
    #     println("Step $step: r = $(exp(trajectoryh[1,1][step].X[2])), th = $(trajectoryh[1,1][step].X[3]), phi = $(trajectoryh[1,1][step].X[4])")
    # end
    trajectoryl = CalculateGeodesics(Xcaml, fovx, fovy, freq, maxnstep, pixels_x, pixels_y, al, Rout, Rstop);
    
    # Integrate the emission along the geodesics
    Imageh = IpoleGeoIntensityIntegration(trajectoryh, freq, pixels_x, pixels_y, scale_factor, ah)
    Imagel = IpoleGeoIntensityIntegration(trajectoryl, freq, pixels_x, pixels_y, scale_factor, al)

    dI_da = (Imageh - Imagel) / (2 * h)  # Finite difference approximation

    return dI_da, Imageh, Imagel
end

function armijo_line_search!(cost_func, compute_gradients!, x, grad, direction, bounds, 
                             args...; α=1e-4, β=0.5, initial_step=1.0, max_steps=10)
    """
    Improved Armijo line search with proper bounds handling and more aggressive stepping
    """
    step_size = initial_step
    x_new = similar(x)
    
    # Initial cost and directional derivative
    f0 = cost_func(x, args...)
    df0 = dot(grad, direction)
    println("grad = $grad, direction = $direction")

    println("  Line search: f0=$f0, df0=$df0, initial_step=$initial_step")
    
    # If direction is not a descent direction, return current point
    if df0 >= 0
        @warn "Not a descent direction, df0 = $df0"
        return x, f0, 0.0
    end
    
    best_x = copy(x)
    best_f = f0
    best_step = 0.0
    
    for i in 1:max_steps
        # Propose new point
        x_new .= x .+ step_size .* direction
        
        # Project onto bounds
        any_bounded = false
        for j in eachindex(x_new)
            old_val = x_new[j]
            x_new[j] = clamp(x_new[j], bounds[j][1], bounds[j][2])
            if x_new[j] != old_val
                any_bounded = true
            end
        end
        
        if any_bounded
            println("  Step $i: Hit bounds, step_size=$step_size, wanted to try a = $((x[2] .+ step_size .* direction[2] )* 0.6), θo = $((x[1] .+ step_size .* direction[1] )* 60)")
            # if a is the one hitting bounds convert the step_size to a proper value, if it's theta, converge it to a proper value
            if x_new[2] == bounds[2][1] || x_new[2] == bounds[2][2]
                step_size = (x_new[2] - x[2]) / direction[2]
            end
            if x_new[1] == bounds[1][1] || x_new[1] == bounds[1][2]
                step_size = (x_new[1] - x[1]) / direction[1]
            end
            #continue
        end
        
        # Check if step is too small. Sometimes we get stuck in a loop of tiny steps
        step_norm = norm(x_new - x)
        if step_norm < 1e-20
            println("  Step $i: Step norm too small ($step_norm), breaking")
            println(" x_new = $x_new, x = $x, step_size = $step_size")
            break
        end
        
        # Evaluate cost at new point
        f_new = cost_func(x_new, args...)
        println("  Step $i: step_size=$step_size, f_new=$f_new, improvement=$(f0-f_new), spin tested = $(x_new[2] * 0.6), θo tested = $(x_new[1] * 60)")
        
        # Keep track of best point found so far
        if f_new < best_f
            best_x .= x_new
            best_f = f_new
            best_step = step_size
        end
        
        # Check Armijo condition. If necessary, change alpha and beta to be more or less strict
        #armijo_threshold = f0 + α * step_size * df0
        armijo_threshold = f0 + α * step_size * df0
        if f_new <= armijo_threshold
            println("  Armijo condition satisfied!")
            println(" New cost function value: $f_new")
            return x_new, f_new, step_size
        end
        
        println("Reducing step size by beta factor: $β, Armijo Condition: $armijo_threshold, $f0, step_size=$step_size, df0=$df0")
        step_size *= β
        
        # If step becomes too small, return best point found
        if abs(x_new[1] - x[1]) < 1e-3 && abs(x_new[2] - x[2]) < 1e-3
            println("  Both parameter changes below 1e-3 threshold, returning best point found")
            return best_x, best_f, best_step
        end
    end
    
    # If no Armijo condition met but we found improvements, return best
    if best_f < f0
        println("  No Armijo satisfaction, but found improvement: $(f0 - best_f)")
        return best_x, best_f, best_step
    end
    
    # If no improvement found at all, return original point
    @warn "Line search failed to find any improvement"
    return x, f0, best_step
end

function true_conjugate_gradient_optimization(Iobs, ro, θoi, ai, freq, nx, ny, nmaxstep, 
                                            fovx, fovy, Rout, Rstop, σ_pixels=0.0; 
                                            cost_tol=2e-11, param_tol=1e-8, grad_tol=1e-10,
                                            max_iterations=200, cg_restart_freq=20,
                                            optimize_param::Symbol=:both)
    """
    True conjugate gradient optimization with proper scaling and convergence criteria
    
    Parameters:
    @Iobs: Observed image intensities (2D array) - This is the one we want to fit
    @ro: Observer radial coordinate
    @θoi: Initial guess for inclination angle in degrees
    @ai: Initial guess for spin parameter (0 to 0.994)
    @freq: Observed frequency, usually 230 GHz
    @nx, @ny: Image dimensions. The resolution of the image
    @nmaxstep: Maximum number of integration steps for geodesics
    @fovx, @fovy: Field of view in radians
    @Rout: Outer radius for geodesic integration
    @Rstop: Stopping radius for geodesic integration
    @σ_pixels: Standard deviation for Gaussian filter applied to intensities and gradients
    @cost_tol: Tolerance for cost function convergence
    @param_tol: Tolerance for parameter change convergence
    @grad_tol: Tolerance for gradient norm convergence
    @max_iterations: Maximum number of optimization iterations
    @cg_restart_freq: Frequency of conjugate gradient restarts
    @optimize_param: Symbol indicating which parameter(s) to optimize (:both, :theta, :spin)

    """
    # Pre-allocate trajectory arrays for each thread (this is going to be used in the autodiff function)
    num_threads = Threads.nthreads()
    thread_trajs = Vector{Vector{OfTraj}}(undef, num_threads)
    for tid in 1:num_threads
        thread_trajs[tid] = Vector{OfTraj}()
        sizehint!(thread_trajs[tid], nmaxstep)
    end

    # You can optimize either both parameters, only θo, or only a, it depends on the optimize_param argument. Make sure it is one of the three options.
    if !(optimize_param in [:both, :theta, :spin])
        throw(ArgumentError("optimize_param must be :both, :theta, or :spin"))
    end
    
    # Parameter scaling factors (normalize to similar ranges)
    θo_scale = 60.0
    a_scale = 0.6
    scales = [θo_scale, a_scale]
    
    # Initialize with scaled parameters
    x_scaled = [θoi / θo_scale, ai / a_scale]
    bounds_scaled = [(0.1 / θo_scale, 90.0 / θo_scale), 
                     (0.0 / a_scale, 0.994 / a_scale)]
    
    # Determine which parameters to optimize
    optimize_theta = optimize_param in [:both, :theta]
    optimize_spin = optimize_param in [:both, :spin]
    
    println("Optimization mode: $optimize_param")
    println("Optimizing θo: $optimize_theta, Optimizing a: $optimize_spin")
    
    dI_dθo = Matrix{Float64}(undef, nx, ny)
    dI_da = Matrix{Float64}(undef, nx, ny)
    I_calc = Matrix{Float64}(undef, nx, ny)
    
    function compute_cost_and_gradients(x_scaled_val, σ_pixels=0.0)
        # Convert back to physical parameters
        θo_val = x_scaled_val[1] * θo_scale
        a_val = x_scaled_val[2] * a_scale
        println("Running AutoDiffGeoTrajEulerMethod with θo = $θo_val, a = $a_val and applying σ_pixels = $σ_pixels filter")
        # Compute intensities and derivatives
        Threads.@threads for i in 0:(nx-1)
            for j in 0:(ny-1)
                tid = Threads.threadid()
                dI_dθo_out = Ref{Float64}()
                intensity_out = Ref{Float64}()
                dI_da_out = Ref{Float64}()
                
                AutoDiffGeoTrajEulerMethod!(thread_trajs[tid],dI_dθo_out, intensity_out, dI_da_out,
                    ro, θo_val, a_val, nx, ny, nmaxstep, i, j, freq, fovx, fovy, Rout, Rstop)
                
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
        
        # Zero out gradients if evolving only theta or only spin. 
        if !optimize_theta
            grad_scaled[1] = 0.0
        end
        if !optimize_spin
            grad_scaled[2] = 0.0
        end
        
        return cost, grad_scaled
    end
    
    function constrained_armijo_line_search!(cost_func, compute_gradients!, x, grad, direction, bounds, args...; kwargs...)
        constrained_direction = copy(direction)
        if !optimize_theta
            constrained_direction[1] = 0.0
        end
        if !optimize_spin
            constrained_direction[2] = 0.0
        end
        
        return armijo_line_search!(cost_func, compute_gradients!, x, grad, constrained_direction, bounds, args...; kwargs...)
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
        grad_converged = false # For now, I'm not using grad convergence, maybe implement this later

        # Relative cost improvement over recent iterations
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
    

    # Initial evaluation
    cost, grad = cached_compute_cost_and_gradients(x_scaled, σ_pixels)
    initial_cost = cost

    if check_convergence(cost, grad, [cost], 0)
        θo_final = x_scaled[1] * θo_scale
        a_final = x_scaled[2] * a_scale
        println("Initial solution already satisfies tolerance")
        return θo_final, a_final, [cost], 1
    end
    # Initialize CG variables
    direction = -copy(grad)
    # Zero out direction for fixed parameters
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
        
        # Line search with constrained direction
        if (iteration == 1)
            aggressive_initial_step = max(3.0, 0.3 / max(norm(grad), 1e-12))
        else
            if(step_size < 0.05/(a_scale * direction[2] ) && optimize_spin)
                step_size = 0.05/(a_scale * direction[2] )
                println("Step size for a is too small, resetting to $step_size")
            end

            if(step_size < 3.0/(θo_scale * direction[1] ) && optimize_theta)
                step_size = 3.0/(θo_scale * direction[1] )
                println("Step size for θo is too small, resetting to $step_size")
            end

            aggressive_initial_step = step_size
            aggressive_initial_step = max(3.0, 0.3 / max(norm(grad), 1e-12))

        end

        println("Trying aggressive initial step: $aggressive_initial_step, set at iteration $iteration")
        println("Cost before line search: $cost")
        cost_comparison = copy(cost)
        x_new, cost_new, step_size = constrained_armijo_line_search!(
            (x_val, args...) -> cached_compute_cost_and_gradients(x_val, σ_pixels)[1],
            (x_val, args...) -> cached_compute_cost_and_gradients(x_val, σ_pixels)[2],
            x_scaled, grad, direction, bounds_scaled,
            α=1e-5, β=0.5, initial_step=aggressive_initial_step, max_steps=15
        )
        
        # Check for sufficient decrease
        println("Line search completed: new cost = $cost_new, step size = $step_size")
        println("Initial cost after line search: $cost")
        absolute_improvement = cost_comparison - cost_new
        relative_improvement = absolute_improvement / max(abs(cost_comparison), 1e-16)
        println("Cost improvement: $absolute_improvement (relative: $relative_improvement)")
        
        if absolute_improvement <= 1e-16 && iteration > 1
            println("No significant improvement in line search, trying steepest descent")
            direction .= -grad
            # Zero out direction for fixed parameters
            if !optimize_theta
                direction[1] = 0.0
            end
            if !optimize_spin
                direction[2] = 0.0
            end
            
            x_new, cost_new, step_size = constrained_armijo_line_search!(
                (x_val, args...) -> cached_compute_cost_and_gradients(x_val, σ_pixels)[1],
                (x_val, args...) -> cached_compute_cost_and_gradients(x_val, σ_pixels)[2],
                x_scaled, grad, direction, bounds_scaled,
                α=1e-5, β=0.5, initial_step=0.01
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
            # Polak-Ribière formula
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
                println("Beta = $beta, Reset to steepest descent (small denominator)")
            end
        end
    end
    
    @warn "Maximum iterations reached without convergence"
    θo_final = x_scaled[1] * θo_scale
    a_final = x_scaled[2] * a_scale
    return θos, as, costs, max_iterations
end



function true_conjugate_gradient_optimization(Iobs, ro, θoi, ai, freq, nx, ny, nmaxstep, 
                                            fovx, fovy, Rout, Rstop, σ_pixels=0.0; 
                                            cost_tol=2e-11, param_tol=1e-8, grad_tol=1e-10,
                                            max_iterations=200, cg_restart_freq=20,
                                            optimize_param::Symbol=:both, simulation_data = nothing)
    """
    True conjugate gradient optimization with proper scaling and convergence criteria
    
    Parameters:
    @Iobs: Observed image intensities (2D array) - This is the one we want to fit
    @ro: Observer radial coordinate
    @θoi: Initial guess for inclination angle in degrees
    @ai: Initial guess for spin parameter (0 to 0.994)
    @freq: Observed frequency, usually 230 GHz
    @nx, @ny: Image dimensions. The resolution of the image
    @nmaxstep: Maximum number of integration steps for geodesics
    @fovx, @fovy: Field of view in radians
    @Rout: Outer radius for geodesic integration
    @Rstop: Stopping radius for geodesic integration
    @σ_pixels: Standard deviation for Gaussian filter applied to intensities and gradients
    @cost_tol: Tolerance for cost function convergence
    @param_tol: Tolerance for parameter change convergence
    @grad_tol: Tolerance for gradient norm convergence
    @max_iterations: Maximum number of optimization iterations
    @cg_restart_freq: Frequency of conjugate gradient restarts
    @optimize_param: Symbol indicating which parameter(s) to optimize (:both, :theta, :spin)

    """
    # Pre-allocate trajectory arrays for each thread (this is going to be used in the autodiff function)
    num_threads = Threads.nthreads()
    thread_trajs = Vector{Vector{OfTraj}}(undef, num_threads)
    for tid in 1:num_threads
        thread_trajs[tid] = Vector{OfTraj}()
        sizehint!(thread_trajs[tid], nmaxstep)
    end

    # You can optimize either both parameters, only θo, or only a, it depends on the optimize_param argument. Make sure it is one of the three options.
    if !(optimize_param in [:both, :theta, :spin])
        throw(ArgumentError("optimize_param must be :both, :theta, or :spin"))
    end
    
    # Parameter scaling factors (normalize to similar ranges)
    θo_scale = 60.0
    a_scale = 0.6
    scales = [θo_scale, a_scale]
    
    # Initialize with scaled parameters
    x_scaled = [θoi / θo_scale, ai / a_scale]
    bounds_scaled = [(0.1 / θo_scale, 90.0 / θo_scale), 
                     (0.0 / a_scale, 0.994 / a_scale)]
    
    # Determine which parameters to optimize
    optimize_theta = optimize_param in [:both, :theta]
    optimize_spin = optimize_param in [:both, :spin]
    
    println("Optimization mode: $optimize_param")
    println("Optimizing θo: $optimize_theta, Optimizing a: $optimize_spin")
    
    dI_dθo = Matrix{Float64}(undef, nx, ny)
    dI_da = Matrix{Float64}(undef, nx, ny)
    I_calc = Matrix{Float64}(undef, nx, ny)
    
    function compute_cost_and_gradients(x_scaled_val, σ_pixels=0.0, simulation_data = nothing)
        # Convert back to physical parameters
        θo_val = x_scaled_val[1] * θo_scale
        a_val = x_scaled_val[2] * a_scale
        println("Running AutoDiffGeoTrajEulerMethod with θo = $θo_val, a = $a_val and applying σ_pixels = $σ_pixels filter")
        # Compute intensities and derivatives
        Threads.@threads for i in 0:(nx-1)
            for j in 0:(ny-1)
                tid = Threads.threadid()
                dI_dθo_out = Ref{Float64}()
                intensity_out = Ref{Float64}()
                dI_da_out = Ref{Float64}()
                
                AutoDiffGeoTrajEulerMethod_GRMHD!(thread_trajs[tid], dI_dθo_out, intensity_out, dI_da_out,
                    ro, θo_val, phi, bhspin, nx, ny, nmaxstep, i, j, freq, fovx, fovy, Rout, Rstop, simulation_data)
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
        
        # Zero out gradients if evolving only theta or only spin. 
        if !optimize_theta
            grad_scaled[1] = 0.0
        end
        if !optimize_spin
            grad_scaled[2] = 0.0
        end
        
        return cost, grad_scaled
    end
    
    function constrained_armijo_line_search!(cost_func, compute_gradients!, x, grad, direction, bounds, args...; kwargs...)
        constrained_direction = copy(direction)
        if !optimize_theta
            constrained_direction[1] = 0.0
        end
        if !optimize_spin
            constrained_direction[2] = 0.0
        end
        
        return armijo_line_search!(cost_func, compute_gradients!, x, grad, constrained_direction, bounds, args...; kwargs...)
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
        grad_converged = false # For now, I'm not using grad convergence, maybe implement this later

        # Relative cost improvement over recent iterations
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

    function cached_compute_cost_and_gradients(x_scaled_val, σ_pixels=0.0, simulation_data = nothing)
        if last_x_computed !== nothing && last_x_computed ≈ x_scaled_val
            println("Using cached computation for x = $x_scaled_val")
            return last_cost, last_grad
        end
        
        cost, grad = compute_cost_and_gradients(x_scaled_val, σ_pixels, simulation_data)
        last_x_computed = copy(x_scaled_val)
        last_cost = cost
        last_grad = copy(grad)
        return cost, grad
    end
    

    # Initial evaluation
    cost, grad = cached_compute_cost_and_gradients(x_scaled, σ_pixels, simulation_data)
    initial_cost = cost

    if check_convergence(cost, grad, [cost], 0)
        θo_final = x_scaled[1] * θo_scale
        a_final = x_scaled[2] * a_scale
        println("Initial solution already satisfies tolerance")
        return θo_final, a_final, [cost], 1
    end
    # Initialize CG variables
    direction = -copy(grad)
    # Zero out direction for fixed parameters
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
        
        # Line search with constrained direction
        if (iteration == 1)
            aggressive_initial_step = max(3.0, 0.3 / max(norm(grad), 1e-12))
            aggressive_initial_step = 0.3/norm(grad)
        else
            if(step_size < 0.05/(a_scale * direction[2] ) && optimize_spin)
                step_size = 0.05/(a_scale * direction[2] )
                println("Step size for a is too small, resetting to $step_size")
            end

            if(step_size < 3.0/(θo_scale * direction[1] ) && optimize_theta)
                step_size = 3.0/(θo_scale * direction[1] )
                println("Step size for θo is too small, resetting to $step_size")
            end

            aggressive_initial_step = step_size
            aggressive_initial_step = max(3.0, 0.3 / max(norm(grad), 1e-12))

        end

        println("Trying aggressive initial step: $aggressive_initial_step, set at iteration $iteration")
        println("Cost before line search: $cost")
        cost_comparison = copy(cost)
        x_new, cost_new, step_size = constrained_armijo_line_search!(
            (x_val, args...) -> cached_compute_cost_and_gradients(x_val, σ_pixels, simulation_data)[1],
            (x_val, args...) -> cached_compute_cost_and_gradients(x_val, σ_pixels, simulation_data)[2],
            x_scaled, grad, direction, bounds_scaled,
            α=1e-5, β=0.5, initial_step=aggressive_initial_step, max_steps=15
        )
        
        # Check for sufficient decrease
        println("Line search completed: new cost = $cost_new, step size = $step_size")
        println("Initial cost after line search: $cost")
        absolute_improvement = cost_comparison - cost_new
        relative_improvement = absolute_improvement / max(abs(cost_comparison), 1e-16)
        println("Cost improvement: $absolute_improvement (relative: $relative_improvement)")
        
        if absolute_improvement <= 1e-16 && iteration > 1
            println("No significant improvement in line search, trying steepest descent")
            direction .= -grad
            # Zero out direction for fixed parameters
            if !optimize_theta
                direction[1] = 0.0
            end
            if !optimize_spin
                direction[2] = 0.0
            end
            
            x_new, cost_new, step_size = constrained_armijo_line_search!(
                (x_val, args...) -> cached_compute_cost_and_gradients(x_val, σ_pixels, simulation_data)[1],
                (x_val, args...) -> cached_compute_cost_and_gradients(x_val, σ_pixels, simulation_data)[2],
                x_scaled, grad, direction, bounds_scaled,
                α=1e-5, β=0.5, initial_step=0.01
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
        _, grad = cached_compute_cost_and_gradients(x_scaled, σ_pixels, simulation_data)
        
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
            # Polak-Ribière formula
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
                println("Reset to steepest descent (small denominator)")
            end
        end
    end
    
    @warn "Maximum iterations reached without convergence"
    θo_final = x_scaled[1] * θo_scale
    a_final = x_scaled[2] * a_scale
    return θos, as, costs, max_iterations
end