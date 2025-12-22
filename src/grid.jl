function Xtoijk_ghost!(X, del)
    # --- i (r) and j (theta) ---
    
    i_logical = trunc(Int, ((X[2] - startx[2]) / dx[2]) - 0.5 + 1000) - 1000
    j_logical = trunc(Int, ((X[3] - startx[3]) / dx[3]) - 0.5 + 1000) - 1000

    i = clamp(i_logical, 0, N1 - 2) # Clamps logical r-index
    j = clamp(j_logical, 0, N2 - 2) # Clamps logical theta-index

    del[2] = (X[2] - ((i + 0.5) * dx[2] + startx[2])) / dx[2]
    del[3] = (X[3] - ((j + 0.5) * dx[3] + startx[3])) / dx[3]

    del[2] = clamp(del[2], 0.0, 1.0)
    del[3] = clamp(del[3], 0.0, 1.0)

    i += 1 # Final 1-based index for i
    j += 1 # Final 1-based index for j

    # --- k (phi) PERIODIC FIX ---
    phi = rem(X[4], cstopx[4])
    if(phi < 0.0)
        phi += cstopx[4]
    end

    # Calculate the logical k-index. This can range from -1 to N3
    k_logical = trunc(Int, ((phi - startx[4]) / dx[4]) - 0.5 + 1000) - 1000
    
    # Calculate the fractional part (delta) using the *unclamped* logical index
    del[4] = (phi - ((k_logical + 0.5) * dx[4] + startx[4])) / dx[4]
    del[4] = clamp(del[4], 0.0, 1.0)

    # Wrap the logical k-index to the valid 0-based range [0, N3-1]
    k_logical_wrapped = mod(k_logical, N3) 
    
    # Convert to 1-based Julia index [1, N3]
    k = k_logical_wrapped + 1

    return i, j, k
end



function X_in_domain(X)
    if(X[2] < cstartx[2] || X[2] > cstopx[2] || X[3] < cstartx[3] || X[3] > cstopx[3])
        return 0
    end
    return 1
end

function ijktoX(i,j,k, X)
    X[2] = startx[2] + (i + 0.5) * dx[2]
    X[3] = startx[3] + (j + 0.5) * dx[3]
    X[4] = startx[4] + (k + 0.5) * dx[4]
    return
end


function interp_scalar(X, data)
    #del::MVec4 = [0.0, 0.0, 0.0, 0.0]
    del = zeros(eltype(X), 4)

    i, j, k = Xtoijk_ghost!(X, del)

    (N1_data, N2_data, N3_data) = size(data) # Get runtime size (e.g., 128, 64, 32)

    ip1 = i + 1
    jp1 = j + 1

    # --- k (phi) PERIODIC FIX ---
    # k is the base index (e.g., 1 to 32)
    kp1 = k + 1
    if kp1 > N3_data
        kp1 = 1  # Wrap around to the first cell
    end
    # --- END FIX ---

    b1 = 1.0 - del[2]
    b2 = 1.0 - del[3]

    # Interpolate in i and j
    interp = data[i, j, k]   * b1 * b2 +
             data[ip1, j, k] * del[2] * b2 +
             data[i, jp1, k] * b1 * del[3] +
             data[ip1, jp1, k] * del[2] * del[3]

    # Interpolate in k (phi)
    # This uses the wrapped kp1, so it correctly interpolates
    # between the last cell (k=32) and the first cell (kp1=1).
    interp = interp * (1.0 - del[4]) + (
             data[i, j, kp1]   * b1 * b2 +
             data[ip1, j, kp1] * del[2] * b2 +
             data[i, jp1, kp1] * b1 * del[3] +
             data[ip1, jp1, kp1] * del[2] * del[3]
    ) * del[4]

    return interp
end


function interp_scalar_time(X, dataA, dataB, tfac)
    vA = interp_scalar(X, dataA)
    if SLOW_LIGHT
        vB = interp_scalar(X, dataB)
        return (tfac) * vA + (1. -tfac) * vB
    end
    return vA
end



function gdet_zone(i,j,k)
    X::MVec4 = MVec4(undef)
    ijktoX(i,j,k, X)
    Xzone::MVec4 = MVec4(undef)
    Xzone[1] = 0.
    Xzone[2] = startx[2] + (i + 0.5) * dx[2]
    Xzone[3] = startx[3] + (j + 0.5) * dx[3]
    Xzone[4] = startx[4] + (k + 0.5) * dx[4]

    gcovKS::MMat4 = MMat4(undef)
    gcov::MMat4 = MMat4(undef)

    for μ in 1:NDIM
        for ν in 1:NDIM
            gcovKS[μ, ν] = 0.
            gcov[μ, ν] = 0.
        end
    end
    rt = MVector{2,Float64}(undef)
    bl_coord!(rt, X)
    r = rt[1]
    th = rt[2]
    gcov_ks(r, th, bhspin, gcovKS)
    dxdX = set_dxdX(Xzone)

    for μ in 1:NDIM
        for ν in 1:NDIM
            for λ in 1:NDIM
                for κ in 1:NDIM
                    gcov[μ, ν] += dxdX[λ, ν] * dxdX[λ, μ] * gcovKS[λ, κ]
                end
            end
        end
    end

    return gdet_func(gcov)

end
