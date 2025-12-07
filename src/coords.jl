
export theta_func


function theta_func(X)
    """
    Computes the theta coordinate from the internal coordinates.
    Parameters:
    @X: Vector of position coordinates in internal coordinates.
    """
    _, th = bl_coord(X)
    return th
end



function bl_to_ks(X, ucon_bl, bhspin)
    """
    Converts the 4-velocity from Boyer-Lindquist coordinates to Kerr-Schild coordinates.
    Parameters:
    @X: Vector of position coordinates in internal coordinates.
    @ucon_bl: Contravariant 4-velocity in Boyer-Lindquist coordinates.
    """
    T = promote_type(eltype(X), eltype(ucon_bl), typeof(bhspin))  # Ensure correct type
    ucon_ks = MVector{4, T}((zero(T), zero(T), zero(T), zero(T)))

    r, th = bl_coord(X)
    trans = MMatrix{4, 4, T}(undef)
    for μ in 1:4
        for ν in 1:4
            trans[μ, ν] = μ == ν ? one(T) : zero(T)
        end
    end

    denom = r * r - 2.0 * r + bhspin * bhspin
    trans[1, 2] = 2.0 * r / denom
    trans[4, 2] = bhspin / denom

    for μ in 1:4
        for ν in 1:4
            ucon_ks[μ] += trans[μ, ν] * ucon_bl[ν]
        end
    end

    return ucon_ks
end




function ks_to_bl(X, ucon_ks, bhspin::Float64)
    """
    Converts the 4-velocity from Kerr-Schild coordinates to Boyer-Lindquist coordinates.
    Parameters:
    @X: Vector of position coordinates in internal coordinates.
    @ucon_ks: Contravariant 4-velocity in Kerr-Schild coordinates.
    """
    ucon_bl = zero(MVec4)
    r, _ = bl_coord(X)

    trans = MMat4(undef)
    for μ in 1:NDIM
        for ν in 1:NDIM
            trans[μ, ν] = μ == ν ? 1.0 : 0.0
        end
    end

    trans[1,2] = 2.0 * r / (r * r - 2.0 * r + bhspin * bhspin)
    trans[4,2] = bhspin / (r * r - 2.0 * r + bhspin * bhspin)

    # Invert the transformation matrix
    rev_trans = inv(trans)

    for μ in 1:NDIM
        ucon_bl[μ] = 0.0
        for ν in 1:NDIM
            ucon_bl[μ] += rev_trans[μ, ν] * ucon_ks[ν]
        end
    end

    return ucon_bl
end
    

    
function vec_to_bl(X::MVec4, v_nat::MVec4, bhspin::Float64)
    """
    Converts a 4-vector from the native coordinate system to Boyer-Lindquist coordinates.
    Parameters:
    @X: Vector of position coordinates in internal coordinates.
    @v_nat: 4-vector in the native coordinate system.
    """

    #First, convert the vector to Kerr-Schild coordinates
    v_ks = zero(MVec4)
    dxdX = MMat4(undef)
    dxdX = set_dxdX(X)
    for μ in 1:NDIM
        for ν in 1:NDIM
            v_ks[μ] += dxdX[μ, ν] * v_nat[ν]
        end
    end

    #Now, convert the Kerr-Schild vector to Boyer-Lindquist coordinates
    vec_bl = zero(MVec4)
    r, _ = bl_coord(X)
    trans = MMat4(undef)
    for μ in 1:NDIM
        for ν in 1:NDIM
            trans[μ, ν] = if μ == ν 1.0 else 0.0 end
        end
    end
    trans[1,2] = 2.0 * r / (r * r - 2.0 * r + bhspin * bhspin)
    trans[4,2] = bhspin / (r * r - 2.0 * r + bhspin * bhspin)

    # Invert the transformation matrix
    rev_trans = inv(trans)

    for μ in 1:NDIM
        ucon_bl[μ] = 0.0
        for ν in 1:NDIM
            vec_bl[μ] += rev_trans[μ, ν] * v_ks[ν]
        end
    end

    return vec_bl
end

function vec_to_ks(X::MVec4, v_nat::MVec4)
    """
    Converts a 4-vector from the native coordinate system to Kerr-Schild coordinates.
    Parameters:
    @X: Vector of position coordinates in internal coordinates.
    @v_nat: 4-vector in the native coordinate system.
    """
    v_ks = zero(MVec4)
    dxdX = MMat4(undef)
    dxdX = set_dxdX(X)

    for μ in 1:NDIM
        for ν in 1:NDIM
            v_ks[μ] += dxdX[μ, ν] * v_nat[ν]
        end
    end

    return v_ks
end

function set_dxdX(X)
    """
    Computes the Jacobian matrix dxdX for the transformation from Kerr-Schild coordinates to internal coordinates.
    Parameters:
    @X: Vector of position coordinates in internal coordinates.
    """
    T = eltype(X)
    dxdX = TMMat4{T}(undef)
    for mu in 1:NDIM
        for nu in 1:NDIM
            if mu == nu
                dxdX[mu, nu] = 1.0
            else
                dxdX[mu, nu] = 0.0
            end
        end
    end

    dxdX[2,2] = exp(X[2])
    if(MODEL == "analytic" || MODEL == "thin_disk")
        dxdX[3,3] = π
    elseif(MODEL == "iharm")
        if(METRIC == "MKS")
            dxdX[3,3] = π + (1 - hslope) * π * cos(2*π*X[3])
        elseif(METRIC == "FMKS")
            dxdX[3, 2] = -exp(mks_smooth * (startx[2] - X[2])) * mks_smooth * (π/2 - π*X[3] +poly_norm * (2*X[3] - 1) * (1 + ((-1 + 2*X[3])/poly_xt)^poly_alpha / (1 + poly_alpha)) - 0.5 * (1 - hslope) * sin(2*π*X[3]))
            dxdX[3, 3] = π + (1 - hslope) * π * cos(2*π*X[3]) +exp(mks_smooth * (startx[2] - X[2])) * (-π +2 * poly_norm * (1 + ((2*X[3]-1)/poly_xt)^poly_alpha / (poly_alpha + 1)) +(2 * poly_alpha * poly_norm * (2*X[3]-1) * ((2*X[3]-1)/poly_xt)^(poly_alpha-1)) / ((1 + poly_alpha) * poly_xt) -(1 - hslope) * π * cos(2*π*X[3]))
        else
            error("Unknown METRIC type: $METRIC")
        end
    end
    if(dxdX[3,3] <= 0.0)
        println("Warning! dxdX[3,3] is non-positive: ", dxdX[3,3])
        println("X[3] = ", X[3])
        dxdX[3,3] = 1.0e-10  # Set a small positive value to avoid issues
    end

    return dxdX
end

function gcov_ks(r, th, bhspin, gcov)
    cth = cos(th)
    sth = sin(th)

    s2 = sth * sth
    rho2 = r * r + bhspin * bhspin * cth * cth

    gcov[1,1] = -1. + 2. * r/rho2
    gcov[1,2] = 2. * r/rho2
    gcov[1,4] = -2. * bhspin * r * s2 / rho2

    gcov[2,1] = gcov[1,2]
    gcov[2,2] = 1. + 2. * r/rho2
    gcov[2,4] = -bhspin * s2 * (1. + 2. * r/rho2)

    gcov[3,3] = rho2

    gcov[4,1] = gcov[1,4]
    gcov[4,2] = gcov[2,4]
    gcov[4,4] = s2 * (rho2 + bhspin * bhspin * s2 * (1. + 2. * r/rho2))
end

function set_dXdx(X)
    """
    Computes the inverse Jacobian matrix dXdx for the transformation from internal coordinates to Kerr-Schild coordinates.
    Parameters:
    @X: Vector of position coordinates in internal coordinates.
    """
    T = eltype(X)
    dxdX = TMMat4{T}(undef)
    dxdX = set_dxdX(X)
    #invert matrix to find dXdx from dxdX using linear algebra package
    dXdx = inv(dxdX)

    return dXdx
end


function vec_from_ks(X, v_ks)
    """
    Converts a 4-vector from Kerr-Schild coordinates to the native coordinate system.
    Parameters:
    @X: Vector of position coordinates in internal coordinates.
    @v_ks: 4-vector in Kerr-Schild coordinates.
    """
    v_nat = zero(TMVec4{eltype(v_ks)})
    T = eltype(X)
    dXdx = TMMat4{T}(undef)
    dXdx = set_dXdx(X)

    for μ in 1:NDIM
        for ν in 1:NDIM
            v_nat[μ] += dXdx[μ, ν] * v_ks[ν]
        end
    end

    return v_nat
end


function bl_coord(X, R0::Float64 = 0.0)
    """
    Returns Boyer-Lindquist coordinates (r, th) from internal coordinates (X[2], X[3]).
    Parameters:
    @X: Vector of position coordinates in internal coordinates coordinates.
    """
    r = exp(X[2]) + R0;

    if(MODEL == "analytic" || MODEL == "thin_disk")
        th = π *X[3]
    elseif(MODEL == "iharm")
        if(METRIC == "FMKS")
            thG = π * X[3] + ((1. - hslope) / 2.) * sin(2. * π * X[3]);
            y = 2 * X[3] - 1.;
            thJ = poly_norm * y* (1. + ((y / poly_xt)^poly_alpha) / (poly_alpha + 1.)) + 0.5 * π;
            th = thG + exp(mks_smooth * (startx[2] - X[2])) * (thJ - thG); 
        elseif(METRIC == "MKS")
            th = π * X[3] + ((1. - hslope) / 2.) * sin(2. * π * X[3]);
        else
            error("Unknown METRIC type: $METRIC")
        end
    end
    return r, th
end

function bl_coord!(rt, X, R0::Float64 = 0.0)
    """
    Returns Boyer-Lindquist coordinates (r, th) from internal coordinates (X[2], X[3]).
    Parameters:
    @X: Vector of position coordinates in internal coordinates coordinates.
    """
    rt[1] = exp(X[2]) + R0;

    if(MODEL == "analytic" || MODEL == "thin_disk")
        rt[2] = π *X[3]
    elseif(MODEL == "iharm")
        if(METRIC == "FMKS")
            thG = π * X[3] + ((1. - hslope) / 2.) * sin(2. * π * X[3]);
            y = 2 * X[3] - 1.;
            thJ = poly_norm * y* (1. + ((y / poly_xt)^poly_alpha) / (poly_alpha + 1.)) + 0.5 * π;
            rt[2] = thG + exp(mks_smooth * (startx[2] - X[2])) * (thJ - thG); 
        elseif(METRIC == "MKS")
            rt[2] = π * X[3] + ((1. - hslope) / 2.) * sin(2. * π * X[3]);
        else
            error("Unknown METRIC type: $METRIC")
        end
    end
end

function flip_index(vector, metric)
    """
    Returns the flipped index of a vector using the metric tensor.
    
    Parameters:
    @vector: Vector to be flipped.
    @metric: Metric tensor used for flipping.
    """
    flipped_vector = zero(TMVec4{eltype(metric)})
    for ν in 1:NDIM
        for μ in 1:NDIM
            flipped_vector[ν] += metric[ν, μ] * vector[μ]
        end
    end
    return flipped_vector
end

@inline function flip_index!(flipped_vector, vector, metric)
    @inbounds for ν in 1:NDIM
        s = zero(eltype(vector))
        for μ in 1:NDIM
            s += metric[ν, μ] * vector[μ]
        end
        flipped_vector[ν] = s
    end
end
