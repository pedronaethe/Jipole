
using Statistics
using ForwardDiff: Dual
# Add these helper structs and functions OUTSIDE your main function (at module level):

# Create callable structs to avoid closure allocations
struct RadTransferX{T1,T2,T3,T4,T5}
    Kconi::T1
    freq::T2
    Intensity::T3
    bhspin::T4
    data::T5
end
(f::RadTransferX)(x) = RadTransferDiff(x, f.Kconi, f.freq, f.Intensity, f.bhspin, f.data)

struct RadTransferK{T1,T2,T3,T4,T5}  
    Xi::T1
    freq::T2
    Intensity::T3
    bhspin::T4
    data::T5
end
(f::RadTransferK)(k) = RadTransferDiff(f.Xi, k, f.freq, f.Intensity, f.bhspin, f.data)

struct RadTransferA{T1,T2,T3,T4,T5}
    Xi::T1
    Kconi::T2  
    freq::T3
    Intensity::T4
    data::T5
end
(f::RadTransferA)(spin) = RadTransferDiff(f.Xi, f.Kconi, f.freq, f.Intensity, spin, f.data)

struct RadTransferI{T1,T2,T3,T4,T5}
    Xi::T1
    Kconi::T2
    freq::T3  
    bhspin::T4
    data::T5
end
(f::RadTransferI)(intens) = RadTransferDiff(f.Xi, f.Kconi, f.freq, intens, f.bhspin, f.data)

function Mom4ODE(X::AbstractVector, Kcon::AbstractVector, bhspin)
    T = eltype(Kcon)
    lconn = TTensor3D{T}(undef)
    get_connection_analytic!(X, lconn, bhspin)
    result = MVector{4, eltype(Kcon)}(undef)
    result .= 0 
    for mu in 1:4
        for alpha in 1:4
            for beta in 1:4
                result[mu] += lconn[mu, alpha, beta] * Kcon[alpha] * Kcon[beta]
            end
        end
    end
    return -result
end

function systemODEs_flat(XK)
    #XK[1:4] = x^μ
    #XK[5:8] = K^μ
    #XK[9] = bhspin
    @views return Mom4ODE(XK[1:4], XK[5:8], XK[9])
end

function CalculateK(ro, θo, i,j, nx, ny, fovx, fovy, bhspin, freq, Rout)
    Xcam = camera_position(ro, θo, 0.0, bhspin, Rout)
    T = eltype(Xcam)
    Kcon = MVector{4, T}(undef)
    X = MVector{4, T}(undef)
    init_XK!(X, Kcon, i, j, Xcam, nx, ny, fovx, fovy, bhspin)
    return Kcon* freq * HPL / (ME * CL * CL)
end




function RadTransferDiff(Xi, Kconi, freq, Ii, bhspin, data)
    # Check this later on
    ji, ki = get_jk(Xi, Kconi, freq, bhspin, data)
    return ji - ki * Ii
end

function AutoDiffGeoTrajEulerMethod!(traj, dI_dθo_out::Base.RefValue{Float64}, intensity_out::Base.RefValue{Float64}, dI_da_out::Base.RefValue{Float64},ro::Float64, θo::Float64, bhspin::Float64, nx::Int64, ny::Int64, nmaxstep::Int64,i::Int64,j::Int64,freq::Float64, fovx::Float64, fovy::Float64, Rout::Float64, Rstop::Float64, data = nothing)
    """
    Returns the intensity and the derivative of the intensity with respect to θo for pixel (i,j) using autodiff.

    Parameters:
    @ro: Distance of the camera in Rg.
    @θo: Angle of the camera in degrees.
    @bhspin: Spin of the black hole.
    @nx: Number of pixels in the x direction.
    @ny: Number of pixels in the y direction.
    @nmaxstep: Maximum number of steps for the geodesic integration.
    @i: Pixel index in the x direction.
    @j: Pixel index in the y direction.
    @freq: Frequency of the radiation.
    @scalefactor: Scale factor for the intensity.
    @fovx: Field of view in the x direction.
    @fovy: Field of view in the y direction.

    Observations:
    - At first it does the geodesic integration using the RK2 method. In the geodesic integration step, it also calculates dX_dθo and dK_dθo, which are the derivatives of the position and momentum with respect to θo.
    - After the geodesic integration, it integrates the intensity along the geodesics using the approximate_solve function. There, it also calculates the derivative of the intensity with respect to θo.
    """


    #=========================================================USED FOR GEODESIC INTEGRATION=========================================================#
    #First set up the initial position and momentum of the specific pixel (i,j)
    Xcam = MVec4(camera_position(ro, θo, 0.0, bhspin, Rout))
    Kcon = MVec4(undef)
    X = MVec4(undef)
    Rh = 1 + sqrt(1. - bhspin * bhspin);  # Radius of the horizon

    #Define X and Kcon
    init_XK!(X, Kcon, i,j, Xcam, nx, ny, fovx, fovy, bhspin)
    #Put Kcon in correct unitless
    Kcon .*= freq * HPL / (ME * CL * CL) 
    dl_unit::Float64 = L_unit * HPL / (ME * CL^2)  # Unit conversion factor for dl

    # half steps, used for polarization
    # Set the variables before (i) and after (f) the first step
    Xhalf = copy(X)
    Kconhalf = copy(Kcon)
    lconn = Tensor3D(undef)

    #Calculate the derivative of the initial positions and momentum with respect to θo
    #The derivative of K is calculated using finite differences
    #Define reference for the intensity integration part so that I don't have to reallocate each step
    jac = MMatrix{4, 9, Float64}(undef)
    if(MODEL == "analytic" || MODEL == "thin_disk")
        dX_dθo = ForwardDiff.derivative(x -> camera_position(ro, x, 0.0, bhspin, Rout), θo)
        dK_dθo = ForwardDiff.derivative(x -> CalculateK(ro, x, i, j, nx, ny, fovx, fovy, bhspin, freq, Rout), θo)

    elseif (MODEL == "iharm")
        # Use finite differences for iharm model
        # iharm model uses bisection method to find camera theta coordinate, which is not differentiable
        # because of the if statements inside the root finding
        del = 1e-6
        θl = θo - del
        θh = θo + del
        Xl = camera_position(ro, θl, 0.0, bhspin, Rout)
        Xh = camera_position(ro, θh, 0.0, bhspin, Rout)
        dX_dθo = (Xh .- Xl) ./ (2.0 * del)
        Kl = CalculateK(ro, θl, i, j, nx, ny, fovx, fovy, bhspin, freq, Rout)
        Kh = CalculateK(ro, θh, i, j, nx, ny, fovx, fovy, bhspin, freq, Rout)
        dK_dθo = (Kh .- Kl) ./ (2.0 * del)
        # dX_dθo = ForwardDiff.derivative(x -> camera_position(ro, x, 0.0, bhspin, Rout), θo)
        # dK_dθo = ForwardDiff.derivative(x -> CalculateK(ro, x, i, j, nx, ny, fovx, fovy, bhspin, freq, Rout), θo)
    end 

    dX_da = ForwardDiff.derivative(x -> camera_position(ro, θo, 0.0, x, Rout), bhspin)
    dK_da = ForwardDiff.derivative(x -> CalculateK(ro, θo, i, j, nx, ny, fovx, fovy, x, freq, Rout), bhspin)

    XK = MVector{9, Float64}(undef)
    XK[9] = bhspin
    #push first step to trajectory

    insert!(traj, 1, OfTraj(
    0,
    copy(X),
    copy(Kcon),
    copy(Xhalf),
    copy(Kconhalf),
    copy(dX_dθo),
    copy(dK_dθo),
    copy(dX_da),
    copy(dK_da)
    ))


    
    #Check if it has been saved in traj.
    # Continue from the second step, since first step is defined at init_XK!
    step::Int64 = 1


    # Pre-allocate temporary vectors to avoid allocations in the main loop
    temp_dX_dθo = MVec4(undef)
    temp_dX_da = MVec4(undef)
    temp_dK_dθo = MVec4(undef) 
    temp_dK_da = MVec4(undef)

    # Also pre-allocate for matrix-vector operations
    temp_jac_dX_dθo = MVec4(undef)
    temp_jac_dK_dθo = MVec4(undef)
    temp_jac_dX_da = MVec4(undef)
    temp_jac_dK_da = MVec4(undef)
    cstartx = MVec4(0.0, log(Rh), 0.0, 0.0)
    cstopx = MVec4(0.0, log(Rout), 1.0, 2.0 * π)
    while (stop_backward_integration(X, Kcon, Rh, Rstop) == 0 && (step <= nmaxstep))
        @inbounds begin
            #Unite X and Kcon into a single vector for ForwardDiff
            @inbounds for k = 1:4
                XK[k] = X[k]
                XK[k+4] = Kcon[k]
            end

            #Calculate the Jacobian of the system of ODEs with respect to the X^μ, K^μ and spin
            # jac is a 4×9 matrix:
            # Rows: Output variables (dK₁/dλ, dK₂/dλ, dK₃/dλ, dK₄/dλ)
            # Columns: Input variables (1:4 = X₁, X₂, X₃, X₄; 5:8 = K₁, K₂, K₃, K₄; 9 = bhspin)
            # Entry (i, j): ∂(ODE_i)/∂(var_j)
            # Table structure:
            #         | ∂(dK₁/dλ)/∂X₁ ... ∂(dK₁/dλ)/∂X₄ ∂(dK₁/dλ)/∂K₁ ... ∂(dK₁/dλ)/∂K₄ ∂(dK₁/dλ)/∂a |
            #         | ∂(dK₂/dλ)/∂X₁ ... ∂(dK₂/dλ)/∂X₄ ∂(dK₂/dλ)/∂K₁ ... ∂(dK₂/dλ)/∂K₄ ∂(dK₂/dλ)/∂a |
            #         | ∂(dK₃/dλ)/∂X₁ ... ∂(dK₃/dλ)/∂X₄ ∂(dK₃/dλ)/∂K₁ ... ∂(dK₃/dλ)/∂K₄ ∂(dK₃/dλ)/∂a |
            #         | ∂(dK₄/dλ)/∂X₁ ... ∂(dK₄/dλ)/∂X₄ ∂(dK₄/dλ)/∂K₁ ... ∂(dK₄/dλ)/∂K₄ ∂(dK₄/dλ)/∂a |
            ForwardDiff.jacobian!(jac, systemODEs_flat, XK)

            dl = stepsize(X, Kcon, cstartx, cstopx)
            traj[step].dl = dl * dl_unit
            @. temp_dX_dθo = traj[step].dX_dθo - dl * traj[step].dK_dθo
            @. temp_dX_da = traj[step].dX_da - dl * traj[step].dK_da

            mul!(temp_jac_dX_dθo, view(jac, 1:4, 1:4), traj[step].dX_dθo)
            mul!(temp_jac_dK_dθo, view(jac, 1:4, 5:8), traj[step].dK_dθo)
            @. temp_dK_dθo = traj[step].dK_dθo - dl * (temp_jac_dX_dθo + temp_jac_dK_dθo)

            mul!(temp_jac_dX_da, view(jac, 1:4, 1:4), traj[step].dX_da)  
            mul!(temp_jac_dK_da, view(jac, 1:4, 5:8), traj[step].dK_da)
            # Handle the jacobian column separately to avoid broadcasting issues
            @. temp_dK_da = traj[step].dK_da - dl * (temp_jac_dX_da + temp_jac_dK_da)
            # Extract column 9 as a proper vector, not a view
            for k in 1:4
                temp_dK_da[k] = temp_dK_da[k] - dl * jac[k, 9]
            end

            push_photon!(X, Kcon, -dl,Xhalf, Kconhalf, lconn, bhspin)
            # if(i == 0 && j == 0)
            #     println("step = $step")
            #     println("dX_dθo = $temp_dX_dθo")
            #     println("dK_dθo = $temp_dK_dθo")
            # end
            # println("$i, $j, $step, $dX_dθo")
            # println("X = $X, stop_backward_integration = $(stop_backward_integration(X, Kcon, Rh, Rstop))")
            step += 1
            insert!(traj, step ,OfTraj(
                copy(dl),
                copy(X),   
                copy(Kcon),   
                copy(Xhalf),   
                copy(Kconhalf),
                copy(temp_dX_dθo),
                copy(temp_dK_dθo),
                copy(temp_dX_da),
                copy(temp_dK_da)
            ))
        end
    end

    if (step > nmaxstep)
        @error("AutoDiffGeoTrajEulerMethod: Maximum number of steps reached without meeting geodesics stop condition.")
        error()
    end
    resize!(traj, length(traj)) 
    # Determine the correct Dual type from autodiff

    #=========================================================USED FOR INTENSITY INTEGRATION=========================================================#
    # Integrate intensity forward
    Xi = MVec4(undef)
    Kconi = MVec4(undef)
    Xf = MVec4(undef)
    Kconf = MVec4(undef)
    Intensity = 0.0
    dI_dθo = 0.0
    dI_da = 0.0
    jac_I_X = MVec4(undef)
    jac_I_K = MVec4(undef)

    for k in 1:NDIM
        Xi[k] = traj[step].X[k]
        Kconi[k] = traj[step].Kcon[k]
    end

    ji, ki = get_jk(Xi, Kconi, freq, bhspin, data)

    # Then replace your ForwardDiff calls in the loop with:
    for nstep = step:-1:2
        for k in 1:NDIM
            Xi[k] = traj[nstep].X[k]
            Xf[k] = traj[nstep - 1].X[k]
            Kconi[k] = traj[nstep].Kcon[k]
            Kconf[k] = traj[nstep - 1].Kcon[k]
        end

        if(MODEL == "thin_disk")
            if(thindisk_region(Xi, Xf))
                Intensity = GetTDBoundaryCondition(Xi, Kconi, bhspin, Rh)
            end
            continue
        end

        if !radiating_region(Xf, Rh)
            continue
        end

        # Update the callable structs with current values
        rad_x = RadTransferX(Kconi, freq, Intensity, bhspin, data)
        rad_k = RadTransferK(Xi, freq, Intensity, bhspin, data)
        rad_a = RadTransferA(Xi, Kconi, freq, Intensity, data)
        rad_i = RadTransferI(Xi, Kconi, freq, bhspin, data)

        # Use the callable structs instead of closures
        ForwardDiff.gradient!(jac_I_X, rad_x, Xi)
        ForwardDiff.gradient!(jac_I_K, rad_k, Kconi)
        jac_I_A = ForwardDiff.derivative(rad_a, bhspin)
        jac_I_I = ForwardDiff.derivative(rad_i, Intensity)

        dI_dθo = dI_dθo + (traj[nstep - 1].dl) * (dot(jac_I_X, traj[nstep].dX_dθo) + dot(jac_I_K, traj[nstep].dK_dθo) + jac_I_I * dI_dθo)
        dI_da = dI_da + (traj[nstep].dl) * (dot(jac_I_X, traj[nstep].dX_da) + dot(jac_I_K, traj[nstep].dK_da) + jac_I_I * dI_da + jac_I_A)

        jf, kf = get_jk(Xf, Kconf, freq, bhspin, data)
        Intensity = approximate_solve(Intensity, ji, ki, jf, kf, traj[nstep - 1].dl)
        if (isnan(Intensity) || isinf(Intensity))
            @error "NaN or Inf encountered in intensity calculation at pixel ($i, $j)"
            println("Intensity = $Intensity, ji = $ji, ki = $ki, jf = $jf, kf = $kf")
            print_vector("Kconf =", Kconf)
            print_vector("Kconi =", Kconi)
            error("NaN or Inf encountered in intensity calculation")
        end
        
        ji = jf
        ki = kf
    end

    dI_dθo_out[] = dI_dθo * freq^3
    intensity_out[] = Intensity * freq^3
    dI_da_out[] = dI_da * freq^3
    empty!(traj)
    return nothing
end
