export camera_position

include("../src/metrics.jl")

function root_find(x, cstartx, cstopx)
    """
    Finds the root of the theta function using a bisection method.
    Parameters:
    @x: Vector of position coordinates in internal coordinates.
    """
    th = x[3]

    xa = zeros(eltype(x), length(x))
    xb = zeros(eltype(x), length(x))
    xc = zeros(eltype(x), length(x))

    xa[2] = log(x[2])
    xa[4] = x[4]
    xb[2] = xa[2]
    xb[4] = xa[4]
    xc[2] = xa[2]
    xc[4] = xa[4]
        
    if x[3] < π / 2.
        xa[3] = cstartx[3]
        xb[3] = (cstopx[3] - cstartx[3]) / 2 + SMALL
    else
        xa[3] = (cstopx[3] - cstartx[3]) / 2 - SMALL
        xb[3] = cstopx[3]
    end

    tol:: Float64 = 1.e-9
    tha = theta_func(xa)
    thb = theta_func(xb)

    if(abs(tha - th) < tol)
        return xa[3]
    elseif(abs(thb - th) < tol)
        return xb[3]
    end

    for i in 1:1000
        xc[3] = 0.5 * (xa[3] + xb[3])
        thc = theta_func(xc)

        if((thc - th) * (thb - th) < 0.)
            xa[3] = xc[3]
        else
            xb[3] = xc[3]
        end

        err = thc - th
        if(abs(err) < tol)
            break
        end
    end

  return xc[3];
end

function root_find_newton(x, cstartx, cstopx)
    """
    Finds the root of the theta function using Newton's method.
    Parameters:
    @x: Vector of position coordinates in internal coordinates.
    """
    # Target value we are trying to match
    th_target = x[3]

    # Initialize working vector
    xc = zeros(eltype(x), length(x))
    
    # Pre-fill constant components (assuming index 2 is log-radius and 4 is generic)
    xc[2] = log(x[2])
    xc[4] = x[4]

    # --- Initial Guess ---
    # We start at the midpoint of the search interval to be safe
    xc[3] = 0.5 * (cstartx[3] + cstopx[3])

    # Settings for Newton's Method
    tol::Float64 = 1.e-16
    max_iter::Int = 100
    epsilon::Float64 = 1.e-7 # Step size for finite difference derivative

    for i in 1:max_iter
        # 1. Evaluate the function at current guess
        # f(theta) = theta_func(theta) - th_target
        current_val = theta_func(xc)
        f_val = current_val - th_target

        # Check convergence
        if abs(f_val) < tol
            return xc[3]
        end

        # 2. Calculate Derivative using Finite Difference
        # f'(theta) ≈ (f(theta + eps) - f(theta)) / eps
        current_theta = xc[3]
        xc[3] = current_theta + epsilon
        val_plus = theta_func(xc)
        
        derivative = (val_plus - current_val) / epsilon

        # Reset xc[3] to current theta for the update step
        xc[3] = current_theta

        # Avoid division by zero
        if abs(derivative) < 1.e-14
            println("Warning: Derivative close to zero, Newton method failed.")
            break
        end

        # 3. Newton Update Step
        # x_new = x_old - f(x) / f'(x)
        xc[3] = xc[3] - (f_val / derivative)
        
        # Optional: Clamp the result to stay within bounds if necessary
        # xc[3] = clamp(xc[3], cstartx[3], cstopx[3])
    end

    return xc[3]
end


function camera_position(cam_dist::Float64, cam_theta_angle, cam_phi_angle::Float64, bhspin, Rout::Float64)
    """
    Computes the camera position in internal coordinates based on the distance and angles.
    Parameters:
    @cam_dist: Radial distance of the camera.
    @cam_theta_angle: Polar angle of the camera in degrees.
    @cam_phi_angle: Azimuthal angle of the camera in degrees.
    """

    if(MODEL == "analytic" || MODEL == "thin_disk")

        T = promote_type(typeof(cam_dist), typeof(cam_theta_angle), typeof(cam_phi_angle), typeof(bhspin))
        X = zeros(T, 4)

        X[1] = 0.0
        X[2] = log(cam_dist)
        X[3] = cam_theta_angle / 180 
        X[4] = cam_phi_angle/180 * π
        return X
    elseif (MODEL == "iharm")

        T = promote_type(typeof(cam_dist), typeof(cam_theta_angle), typeof(cam_phi_angle), typeof(bhspin))
        X = zeros(T, 4)
        x = [zero(T), T(cam_dist), T(cam_theta_angle)/T(180) * T(π), T(cam_phi_angle)/T(180) * T(π)]
        X[1] = 0.0
        X[2] = log(cam_dist)
        #X[3] = root_find(x, cstartx, cstopx)
        X[3] = root_find_newton(x, cstartx, cstopx)
        #X[3] = cam_theta_angle / 180
        X[4] = cam_phi_angle/180 * π
        return X
    else
        error("Unknown MODEL type: $MODEL")
    end
end

