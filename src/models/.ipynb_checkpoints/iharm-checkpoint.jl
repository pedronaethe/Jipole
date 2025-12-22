using HDF5
using Printf

# --- Constants for Primitives ---
const VALID_PRIMS = ["RHO", "UU", "U1", "U2", "U3", "B1", "B2", "B3"]
const USE_GEODESIC_SIGMACUT = true
const RHO_unit = M_unit / L_unit^3  # Density unit in g/cm^3
const U_unit = RHO_unit * CL^2  # Internal energy density unit in erg
const B_unit = CL * sqrt(4 * π * RHO_unit)  # Magnetic field unit in Gauss

metric_type = "METRIC_FMKS"
if (metric_type == "METRIC_FMKS")
    const poly_xt = 0.82#TODO: prob have to be read from file
    const poly_alpha = 14.0#TODO: prob have to be read from file
    const mks_smooth = 0.5#TODO: prob have to be read from file
    const poly_norm = 0.5 * π * 1. /(1. + 1. /(poly_alpha + 1. )*1. /(poly_xt^poly_alpha));
end

# --- Data Storage Structure ---
"""
    IharmData

A struct to hold the data from an iharm3d simulation dump.
The data is stored in a dictionary where keys are the primitive names
(e.g., "RHO") and values are 3D arrays.
"""
# struct IharmData
#     primitives::Dict{String, Array{Float64, 3}}
# end

struct IharmData
    RHO::Array{Float64,3}
    UU::Array{Float64,3}
    U1::Array{Float64,3}
    U2::Array{Float64,3}
    U3::Array{Float64,3}
    B1::Array{Float64,3}
    B2::Array{Float64,3}
    B3::Array{Float64,3}
    ne::Array{Float64,3}
    b::Array{Float64,3}
    θe::Array{Float64,3}
    sigma::Array{Float64,3}
    beta::Array{Float64,3}
end



# --- Internal Functions ---

function _read_single_primitive(file_handle, prim_name::String)
    if "prims" in keys(file_handle)
        prims = read(file_handle["prims"])
        prim_idx = findfirst(isequal(uppercase(prim_name)), VALID_PRIMS)
        # Returns 'nothing' if the primitive is not found
        return prim_idx === nothing ? nothing : permutedims(prims[prim_idx, :, :, :], (3, 2, 1))
    else
        error("Dataset 'prims' not found in the HDF5 file.")
    end
end


function load_data(filename::String, Nfiles::Int = 1)
    println("Loading data from '$filename' into 'iharm' module...")
    !isfile(filename) && error("File not found: $filename")

    # Temporary variables to store primitives
    rho = uu = u1 = u2 = u3 = b1 = b2 = b3 = nothing

    #Nfiles will be useful when using SLOW_LIGHT later on, for now we just load one file
    data_array = Vector{IharmData}(undef, Nfiles)


    #Rescale mdot
    rescale_factor = 1.;
    target_mdot = 0; #TODO: THIS HAS TO BE READ FROM INPUT OR PLACED SOMEWHERE ELSE
    if(target_mdot > 0)
        println("Resetting M_unit to match target_mdot = $target_mdot")
        current_mdot = Mdot_dump/MdotEdd_dump
        println("... is now $(M_unit * abs(target_mdot / current_mdot))")
        rescale_factor = abs(target_mdot / current_mdot)
        M_unit *= rescale_factor
    end


    h5open(filename, "r") do file
        for n in 1:Nfiles
            Threads.@threads for prim_name in VALID_PRIMS
                data_3d = _read_single_primitive(file, prim_name)
                if data_3d !== nothing
                    data_3d = Float64.(data_3d)  # ensure Float64

                    # Assign to the correct struct field
                    if prim_name == "RHO"
                        rho = data_3d
                        if(size(rho,1) != N1 || size(rho,2) != N2 || size(rho,3) != N3)
                            error("Data dimensions do not match expected grid size N1,N2,N3")
                        end
                    elseif prim_name == "UU"
                        uu = data_3d
                    elseif prim_name == "U1"
                        u1 = data_3d
                    elseif prim_name == "U2"
                        u2 = data_3d
                    elseif prim_name == "U3"
                        u3 = data_3d
                    elseif prim_name == "B1"
                        b1 = data_3d
                    elseif prim_name == "B2"
                        b2 = data_3d
                    elseif prim_name == "B3"
                        b3 = data_3d
                    end
                end
            end
            data_array[n] = IharmData(rho, uu, u1, u2, u3, b1, b2, b3, zeros(size(rho)), zeros(size(rho)), zeros(size(rho)), zeros(size(rho)), zeros(size(rho)))            
        end
    end

    println("Primitives successfully loaded. Dimensions: ", size(rho))
    println("Calculating physical quantities...")
    for n in 1:Nfiles
        Threads.@threads for i in 1:(N1)
            for j in 1:(N2)
                X::MVec4 = zeros(MVec4)
                ijktoX(i-1, j-1, 0, X)
                gcov::MMat4 = zeros(MMat4)
                gcon::MMat4 = zeros(MMat4)
                gcov_func!(X, bhspin, gcov)
                gcon_func!(gcov, gcon)
                g = gdet_zone(i-1, j-1, 0)

                for k in 1:(N3)
                    ijktoX(i-1, j-1, k, X)

                    Ufields = (data_array[n].U1, data_array[n].U2, data_array[n].U3)
                    UdotU = 0.0
                    for l in 1:(NDIM -1)       # l = 1,2,3 corresponds to U1,U2,U3
                        for m in 1:(NDIM -1)
                            UdotU += gcov[l+1, m+1] * Ufields[l][i,j,k] * Ufields[m][i,j,k]
                        end
                    end

                    ufac = sqrt(-1. / gcon[1,1] * (1. + abs(UdotU)))
                    ucon::MVec4 = MVec4(undef)
                    ucon[1] = -ufac * gcon[1,1]

                    for μ in 1:(NDIM-1)
                        ucon[μ + 1] = Ufields[μ][i,j,k] - ufac * gcon[1, μ+1]
                    end
                    
                    ucov::MVec4 = MVec4(undef)
                    flip_index!(ucov, ucon, gcov)
                    udotB = 0.0
                    Bfields = (data_array[n].B1, data_array[n].B2, data_array[n].B3)
                    for l in 1:(NDIM -1)
                        udotB += ucov[l+1] * Bfields[l][i,j,k]
                    end

                    bcon::MVec4 = MVec4(undef)
                    bcon[1] = udotB
                    for μ in 1:(NDIM-1)
                        bcon[μ+1] = (Bfields[μ][i,j,k] + ucon[μ+1] * udotB) / ucon[1]
                    end
                    bcov = MVec4(undef)
                    flip_index!(bcov, bcon, gcov)

                    bsq = 0.0
                    for l in 1:NDIM
                        bsq += bcov[l] * bcon[l]
                    end
                    data_array[n].b[i,j,k] = sqrt(bsq) * B_unit  # Magnetic field strength

                end
            end
        end
        init_physical_quantities(data_array, n, rescale_factor)  # Initialize physical quantities for the first dataset
    end

    # Check if all fields were loaded
    fields = [rho, uu, u1, u2, u3, b1, b2, b3]
    if any(x -> x === nothing, fields)
        @warn "Some primitives were missing in file '$filename'."
    else
        println("All primitives successfully loaded. Dimensions: ", size(rho))
    end

    return data_array
end


function init_physical_quantities(data, n::Int64, rescale_factor::Float64)
    println("Using mixed tp_over_te with trat_small = $(trat_small), trat_large = $(trat_large), and beta_crit = $(beta_crit)")
    
    # Pre-compute constants
    rescale_factor_sqrt = sqrt(rescale_factor)
    rho_factor = RHO_unit / (MP + ME) * Ne_factor
    gam_minus_1 = gam - 1.0
    beta_crit_sq = beta_crit * beta_crit
    θe_factor = (MP / ME) * (game - 1.0) * (gamp - 1.0)
    game_minus_1 = game - 1.0
    gamp_minus_1 = gamp - 1.0
    B_unit_inv = 1.0 / B_unit
    
    # Get array references once (helps with type stability)
    ne_arr = data[n].ne
    b_arr = data[n].b
    θe_arr = data[n].θe
    sigma_arr = data[n].sigma
    beta_arr = data[n].beta
    RHO_arr = data[n].RHO
    UU_arr = data[n].UU
    
    @inbounds Threads.@threads for i in 1:N1
        for j in 1:N2
            for k in 1:N3
                rho_ijk = RHO_arr[i, j, k]
                uu_ijk = UU_arr[i, j, k]
                b_ijk = b_arr[i, j, k]
                
                ne_arr[i, j, k] = rho_ijk * rho_factor
                
                b_ijk *= rescale_factor_sqrt
                b_arr[i, j, k] = b_ijk
                
                bsq_normalized = b_ijk * B_unit_inv
                bsq = bsq_normalized * bsq_normalized
                
                sigma_m = bsq / rho_ijk
                beta_m = uu_ijk * gam_minus_1 / (0.5 * bsq)
                
                betasq = beta_m * beta_m / beta_crit_sq
                betasq_plus_1_inv = 1.0 / (1.0 + betasq)
                trat = trat_large * betasq * betasq_plus_1_inv + trat_small * betasq_plus_1_inv
                
                θe_unit = θe_factor / (game_minus_1 * trat + gamp_minus_1)
                θe_val = θe_unit * uu_ijk / rho_ijk
                
                θe_arr[i, j, k] = θe_val > 1.0e-3 ? θe_val : 1.0e-3
                sigma_arr[i, j, k] = sigma_m > SMALL ? sigma_m : SMALL
                beta_arr[i, j, k] = beta_m > SMALL ? beta_m : SMALL
            end
        end
    end
end

function get_model_sigma(X, data)
    if (X_in_domain(X) == 0)
        return 0.0
    end
    tfac = 0.0 #TODO: when using slowlight, we should implement this
    nA = 1
    nB = 1

    return interp_scalar_time(X, data[nA].sigma, data[nB].sigma, tfac)
end


function get_sigma_smoothfac(sigma)
    sigma_above = sigma_cut
    if(sigma_cut_high > 0.0)
        sigma_above = sigma_cut_high
    end
    if(sigma < sigma_cut)
        return 1.0
    end
    if(sigma > sigma_above)
        return 0.0
    end
    dsig = sigma_above - sigma_cut
    return cos(π/2. /dsig * (sigma - sigma_cut))
end



function get_model_ne(X, data)
    if(X_in_domain(X) == 0)
        return 0.0
    end
    sigma_smoothfac = 1.0;

    if(USE_GEODESIC_SIGMACUT)
        sigma = get_model_sigma(X, data);
        if(sigma > sigma_cut)
            return 0.0
        end
        sigma_smoothfac = get_sigma_smoothfac(sigma)
    end

    nA = 1
    nB = 1
    tfac = 0.0 #TODO: when using slowlight, we should implement this
    return interp_scalar_time(X, data[nA].ne, data[nB].ne, tfac) * sigma_smoothfac
end

function set_tinterp_ns(X)
    """
    How far have we interpolated in time between two data points data[nA] and data[nB]

    Parameters:
    @X: The point in spacetime where we want to interpolate.

    Observations:
    - In slowlight mode, we perform linear interpolation in time. This function tells
    us how far we've progressed from data[nA]->t to data[nB]->t but "in reverse" as
    tinterp == 1 -> we're exactly on nA and tinterp == 0 -> we're exactly on nB. 
    
    - Currently, this function is just a placeholder, it must be implemented if SLOW_LIGHT is true.
    """

    return 0.0, 0, 0
end

function get_model_thetae(X, data)
    if(X_in_domain(X) == 0)
        return 0.0
    end
    nA = 1
    nB = 1
    tfac = 0.0 #TODO: when using slowlight, we should implement this

    return interp_scalar_time(X, data[nA].θe, data[nB].θe, tfac)
end

function get_model_b(X, data)
    if(X_in_domain(X) == 0)
        return 0.0
    end
    nA = 1
    nB = 1
    tfac = 0.0 #TODO: when using slowlight, we should implement this

    return interp_scalar_time(X, data[nA].b, data[nB].b, tfac)
end

function get_model_fourv(data, X, Kcon, Ucon, Ucov, Bcon, Bcov, bhspin)
    gcov = gcov_func(X, bhspin)
    gcon = gcon_func(gcov)

    if(X_in_domain(X) == 0)
        Ucov[1] = -1. /sqrt(-gcon[1, 1])
        Ucov[2] = 0.0
        Ucov[3] = 0.0
        Ucov[4] = 0.0
        Ucon[1] = 0.0
        Ucon[2] = 0.0
        Ucon[3] = 0.0
        Ucon[4] = 0.0

        for μ in 1:NDIM
            Ucon[1] += Ucov[μ] * gcon[1, μ]
            Ucon[2] += Ucov[μ] * gcon[2, μ]
            Ucon[3] += Ucov[μ] * gcon[3, μ]
            Ucon[4] += Ucov[μ] * gcon[4, μ]
            Bcon[μ] = 0.0
            Bcov[μ] = 0.0
        end
        return
    end
    Vcon = typeof(X)(undef)
    tfac, _, _ = set_tinterp_ns(X)
    nA = 1 #TODO: when using slowlight, we should implement this
    nB = 1 #TODO: when using slowlight, we should implement this
    Vcon[2] = interp_scalar_time(X, data[nA].U1, data[nB].U1, tfac);
    Vcon[3] = interp_scalar_time(X, data[nA].U2, data[nB].U2, tfac);
    Vcon[4] = interp_scalar_time(X, data[nA].U3, data[nB].U3, tfac);
    VdotV = 0.0
    for μ in 2:NDIM
        for ν in 2:NDIM
            VdotV += gcov[μ, ν] * Vcon[μ] * Vcon[ν]
        end
    end

    Vfac = sqrt(-1. /gcon[1, 1] * (1. + abs(VdotV)))
    Ucon[1] = -Vfac * gcon[1, 1]
    for μ in 2:NDIM
        Ucon[μ] =  Vcon[μ] - Vfac * gcon[1, μ]
    end

    Ucov_local = flip_index(Ucon, gcov)

    #Now set Bcon and get Bcov by lowering it

    Bcon1 = interp_scalar_time(X, data[nA].B1, data[nB].B1, tfac);
    Bcon2 = interp_scalar_time(X, data[nA].B2, data[nB].B2, tfac);
    Bcon3 = interp_scalar_time(X, data[nA].B3, data[nB].B3, tfac);

    Bcon[1] = (Ucov_local[2] * Bcon1 + Ucov_local[3] * Bcon2 + Ucov_local[4] * Bcon3)
    Bcon[2] = (Bcon1 + Ucon[2] * Bcon[1]) / Ucon[1]
    Bcon[3] = (Bcon2 + Ucon[3] * Bcon[1]) / Ucon[1]
    Bcon[4] = (Bcon3 + Ucon[4] * Bcon[1]) / Ucon[1]

    Bcov_local = flip_index(Bcon, gcov)

    for μ in 1:NDIM
        Ucov[μ] = Ucov_local[μ]
        Bcov[μ] = Bcov_local[μ]
    end
end



function radiating_region(X::MVec4, Rh::Float64)
    """
    Checks if the position is within the radiating region.
    Parameters:
    @X: Vector of position coordinates in internal coordinates.
    """
    r, th = bl_coord(X)
    return (r > (rmin_geo) && r < rmax_geo && th < (π - th_beg))
end
