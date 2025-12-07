include("../metrics.jl")

#Model parameters (adjust spin in main.jl)
const A = 1.e6
const α_analytic = -0.0
const height = (100. / 3.0)
const l0 = 1.0


const RHO_unit = 3.e-18
const U_unit = RHO_unit * CL^2  # Internal energy density unit in erg
const B_unit = CL * sqrt(4 * π * RHO_unit)  # Magnetic field unit in Gauss

function radiating_region(X::MVec4, Rh::Float64)
    """
    Checks if the position is within the radiating region.
    Parameters:
    @X: Vector of position coordinates in internal coordinates.
    """
    r, _ = bl_coord(X)
    return (r > (Rh + 0.0001) && r > 1. && r < 1000.0)
end


function get_model_4vel(X, bhspin)
    """
    Computes the 4-velocity of the model from the position vector in internal coordinates.
    Parameters:
    @X: Vector of position coordinates in internal coordinates.

    Observations:
    - This follows the model described in the paper (https://iopscience.iop.org/article/10.3847/1538-4357/ab96c6/pdf).
    """

    r,th = bl_coord(X)
    R = r * sin(th) 
    # #Here, we are considering q = 0.5
    l = l0/(1 + R) * R^(1.5)
    T = promote_type(typeof(r), typeof(th), typeof(bhspin))
    bl_gcov = @MMatrix zeros(T, 4, 4)
    bl_gcon = @MMatrix zeros(T, 4, 4)
    bl_Ucov = @MVector zeros(T, 4)
    bl_Ucon = @MVector zeros(T, 4)
    gcov_bl!(r, th, bhspin, bl_gcov)
    gcon_func!(bl_gcov, bl_gcon)
    
    gcov = gcov_func(X, bhspin)
  
    # Get the normal observer velocity for Ucon/Ucov, in BL coordinates
    ubar = sqrt(-1. / (bl_gcon[1,1] - 2. * bl_gcon[1,4] * l+ bl_gcon[4,4] * l * l))
    bl_Ucov[1] = -ubar
    bl_Ucov[2] = zero(eltype(bl_Ucov))
    bl_Ucov[3] = zero(eltype(bl_Ucov))
    bl_Ucov[4] = l * ubar
    bl_Ucon = flip_index(bl_Ucov, bl_gcon)

    ks_Ucon = bl_to_ks(X, bl_Ucon, bhspin)
    Ucon = vec_from_ks(X, ks_Ucon)
    Ucov = flip_index(Ucon, gcov)

    return Ucov
end


function get_model_ne(X)
    """
    Computes the electron number density from the position vector in internal coordinates.
    Parameters:
    @X: Vector of position coordinates in internal coordinates.
    
    Observations:
    - This follows the model described in the paper (https://iopscience.iop.org/article/10.3847/1538-4357/ab96c6/pdf).
    """

    r, th = bl_coord(X)
    
    n_exp = 0.5 * ((r/10)^2 + (height * cos(th))^2)
    return (n_exp < 200) ? RHO_unit * exp(-n_exp) : 0.0
end

function get_analytic_jk(X, Kcon, freqcgs::Float64, bhspin)
    """
    Computes the emissivity and absorption coefficient for the model at a given position and frequency.
    Parameters:
    @X: Vector of position coordinates in internal coordinates.
    @Kcon: Covariant 4-vector of the photon in internal coordinates.
    @freqcgs: pivotal frequency in cgs units.
    """
    Ne = get_model_ne(X)
    if (Ne <= 0.)
        return 0.0, 0.0
    end
    Ucov= get_model_4vel(X, bhspin)
    ν = get_fluid_nu(Kcon, Ucov)
    if(ν <= 0. || any(isnan(ν)) || any(isinf(ν)))
        println("At X = $X\n Kcon = $Kcon")
        println("Ucov = $Ucov")
        println("Kcon $Kcon")
        error("Frequency must be positive, got ν = $ν")
    end

    jnu_inv = max(Ne * (ν/freqcgs)^(-α_analytic)/ν^2, 0.0)
    knu_inv = max((A * Ne * (ν/freqcgs)^(-(α_analytic + 2.5)) + 1.e-54) * ν, 0.0)

    if(isnan(jnu_inv) || isinf(jnu_inv))
        @error "Invalid jnu_inv computed" jnu_inv
        println("Ne = $Ne, ν = $ν")
        error("Invalid jnu_inv computed: $jnu_inv")
    end
    if(isnan(knu_inv) || isinf(knu_inv))
        @error "Invalid knu_inv computed" knu_inv
        println("Ne = $Ne, ν = $ν")
        error("Invalid knu_inv computed: $knu_inv")
    end
    
    return jnu_inv, knu_inv
end
