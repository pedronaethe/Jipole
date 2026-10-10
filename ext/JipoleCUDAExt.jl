"""
CUDA backend for Jipole. Julia loads this module automatically once both `Jipole` and
`CUDA` are loaded, so CPU-only runs never pay for loading CUDA.jl.
"""
module JipoleCUDAExt

using CUDA
using StaticArrays
using Jipole
using Jipole: Constants, Camera, Geodesics, Radiation, Iharm, Imaging, Slowlight, Utils_GPU, Tetrads
using Jipole.GeoTypes: OfTrajGeneric, GPUTrajStep
using Jipole.Slowlight: OfSlowLight, pack_trajectory_tile!

function Utils_GPU.copy_iharm_to_gpu(cpu_data)
    return Iharm.IharmData(
        Float64(cpu_data.t),
        CuArray(cpu_data.RHO),
        CuArray(cpu_data.UU),
        CuArray(cpu_data.U1),
        CuArray(cpu_data.U2),
        CuArray(cpu_data.U3),
        CuArray(cpu_data.B1),
        CuArray(cpu_data.B2),
        CuArray(cpu_data.B3),
        CuArray(cpu_data.ne),
        CuArray(cpu_data.b),
        CuArray(cpu_data.θe),
        CuArray(cpu_data.sigma),
        CuArray(cpu_data.beta),
        CuArray(cpu_data.dθedRhi)
    )
end


"""
    raytrace_image_gpu!(d_traj, d_Image, d_truncated, i_offset, j_offset, block_size_x, block_size_y,
        Xcam, Econ, bhspin, nx, ny, nmaxstep, freq, fovx, fovy, Rout, Rstop, data, params)

GPU kernel launcher for [`calculate_image!`](@ref): raytraces and
integrates the emission for one tile of the image plane (the tile given by
`i_offset`/`j_offset`, of size `block_size_x`×`block_size_y`), writing the
resulting intensities into `d_Image`. Call once per tile from a
`for`-loop over the full image, since `d_traj`'s size limits how much of
the image a single launch can cover.

The division in tiles is necessary depending on the size of the image due to GPU memory constraints.

# Arguments
- `d_traj`: Pre-allocated `CuArray{GPUTrajStep{T}}` scratch buffer, sized
  `(block_size_x, block_size_y, nmaxstep)`.
- `d_Image`: Output image, overwritten in-place.
- `d_truncated`: Pre-allocated array of booleans, set `true` for any pixel whose geodesic
  integration hit `nmaxstep` before `stop_backward_integration` actually
  fired (i.e. `d_traj` was too small for that pixel). Callers should check
  this after the launch and retry with a larger `nmaxstep`/`d_traj` if any
  entry came back `true`.
- `i_offset`, `j_offset`: Pixel offset of this tile within the full image.
- `block_size_x`, `block_size_y`: Tile size (must match `d_traj`'s first
  two dimensions and the launch's thread/block configuration).
- `Xcam`: Camera position in internal coordinates.
- `Econ`: Camera's orthonormal tetrad.
- `bhspin`: Dimensionless black hole spin parameter.
- `nx`, `ny`: Full image resolution.
- `nmaxstep`: Maximum number of geodesic integration steps.
- `freq`: Pivotal frequency, in cgs units.
- `fovx`, `fovy`: Field of view, in radians.
- `Rout`: Outer simulation-grid radius.
- `Rstop`: Backward-integration stopping radius.
- `data`: Model-specific auxiliary data (e.g. `Iharm`'s GRMHD snapshots).
- `params`: Model parameters.
"""
function Imaging.raytrace_image_gpu!(
    d_traj, d_Image, d_truncated,
    i_offset, j_offset, block_size_x, block_size_y, # New offset parameters
    Xcam, Econ, bhspin, nx, ny, nmaxstep,
    freq, fovx, fovy, Rout, Rstop, data, params
)
    local_i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    local_j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    i = local_i - 1 + i_offset
    j = local_j - 1 + j_offset

    if local_i <= block_size_x && local_j <= block_size_y && i < nx && j < ny

        calculate_image!(
            d_traj, d_Image, d_truncated, Xcam, Econ, bhspin, nx, ny, nmaxstep,
            i, j, local_i, local_j, freq, fovx, fovy, Rout, Rstop, params, data
        )
    end
    return nothing
end


"""
    calculate_image!(traj, d_Image, d_truncated, Xcam, Econ, bhspin, nx, ny, nmaxstep, i_global,
        j_global, i_local, j_local, freq, fovx, fovy, Rout, Rstop, params, data=nothing)

GPU per-pixel kernel body for [`raytrace_image_gpu!`](@ref): traces the
geodesic for pixel `(i_global, j_global)` backward from the camera,
storing each step into `traj[i_local, j_local, :]`, then integrates the
radiative transfer equation forward along the stored trajectory (from the
far end back to the camera), writing the result into `d_Image`.

# Arguments
- `traj`: Pre-allocated `CuDeviceArray{GPUTrajStep{T}}` scratch buffer for
  this tile, indexed by the pixel's local (within-tile) coordinates.
- `d_Image`: Output image, overwritten in-place at `(i_global, j_global)`.
- `d_truncated`: Pre-allocated boolean array for this tile, set
  `true` at `(i_local, j_local)` if `nmaxstep` was hit before the geodesic
  actually reached its stopping condition.
- `ro`, `θo`, `phi`: Camera radial distance, inclination, and azimuth.
- `bhspin`: Dimensionless black hole spin parameter.
- `nx`, `ny`: Full image resolution.
- `nmaxstep`: Maximum number of geodesic integration steps.
- `i_global`, `j_global`: Pixel indices in the full image plane.
- `i_local`, `j_local`: Pixel indices within this tile's `traj` buffer.
- `freq`: Pivotal frequency, in cgs units.
- `fovx`, `fovy`: Field of view, in radians.
- `Rout`: Outer simulation-grid radius (unused; kept for call-signature
  symmetry with [`raytrace_image_gpu!`](@ref)).
- `Rstop`: Backward-integration stopping radius.
- `params`: Model parameters.
- `data`: Model-specific auxiliary data (e.g. `Iharm`'s GRMHD snapshots).
"""
function calculate_image!(
    traj, d_Image, d_truncated,
    Xcam::SVector{4,Float64}, Econ::SMatrix{4,4,Float64}, bhspin::Float64,
    nx::Int64, ny::Int64, nmaxstep::Int64,
    i_global::Int64, j_global::Int64,
    i_local::Int64, j_local::Int64, # Accept local matrix indices directly
    freq::Float64, fovx::Float64, fovy::Float64, Rout::Float64, Rstop::Float64, params,
    data::T_data = nothing
) where {T_data}

    if (i_global >= nx || j_global >= ny)
        return nothing
    end

    Kcon0 = Geodesics.init_kcon(i_global, j_global, Econ, nx, ny, fovx, fovy)
    Kcon = Kcon0 * (freq * Constants.HPL / (Constants.ME * Constants.CL * Constants.CL))

    dl_unit::Float64 = params.L_unit * Constants.HPL / (Constants.ME * Constants.CL^2)
    Rh = 1.0 + sqrt(1.0 - bhspin * bhspin)

    X = Xcam
    K = Kcon
    #lconn = MArray{Tuple{4,4,4},Float64,3,64}(undef)

    step::Int64 = 1
    T = eltype(X)
    @inbounds traj[i_local, j_local, step] = GPUTrajStep{T}(
        0.0, X, K
    )
    while (Geodesics.stop_backward_integration(X, K, Rh, Rstop) == 0 && (step < nmaxstep))
        @inbounds begin
            dl = Geodesics.stepsize(X, K, params.cstartx, params.cstopx)
            scaled_dl = dl * dl_unit
            X, K, Xhalf, Khalf = Geodesics.push_photon(X, K, -dl, bhspin, params)
            step += 1
            @inbounds traj[i_local, j_local, step] = GPUTrajStep{T}(
                scaled_dl, X, K
            )
        end
    end

    @inbounds d_truncated[i_local, j_local] = (step >= nmaxstep) && (Geodesics.stop_backward_integration(X, K, Rh, Rstop) == 0)

    # #Radiative Transfer Integration:

    Intensity = 0.0

    @inbounds Xi_S = traj[i_local, j_local, step].X
    @inbounds Kconi_S = traj[i_local, j_local, step].Kcon

    ji, ki = Radiation.get_jk(Xi_S, Kconi_S, freq, bhspin, params, data)

    @inbounds for nstep in step:-1:2
        Xi_S = traj[i_local, j_local, nstep].X
        Xf_S = traj[i_local, j_local, nstep - 1].X
        Kconi_S = traj[i_local, j_local, nstep].Kcon
        Kconf_S = traj[i_local, j_local, nstep - 1].Kcon
        dl_step = traj[i_local, j_local, nstep].dl

        if !Radiation.radiating_region(Xf_S, params, Rh)
            continue
        end
        jf, kf = Radiation.get_jk(Xf_S, Kconf_S, freq, bhspin, params, data)

        Intensity = Radiation.approximate_solve(Intensity, ji, ki, jf, kf, dl_step)

        CUDA.@cuassert !(isnan(Intensity) || isinf(Intensity)) "NaN/Inf Intensity encountered!"

        ji = jf
        ki = kf
    end

    @inbounds d_Image[i_global + 1, j_global + 1] = Intensity * (freq^3)

    return nothing
end



"""
    render_round_gpu!(movie_nstep, movie_intensity, all_geodesics, nsteps, max_nstep, tile_height,
        valid_ks, nimgs_concurrently, target_times, pixels_x, pixels_y, params_slowlight, freq, model, gpu_data)

Render one round on the GPU -- the `:GPU` counterpart of
[`render_round_cpu!`](@ref). Goes tile by tile (rows sized by
[`gpu_tile_plan`](@ref)): pack that tile's trajectories
([`pack_trajectory_tile!`](@ref)) and upload it plus the current
`movie_nstep`/`movie_intensity` state, run [`slowlight_kernel!`](@ref),
then copy the results back to `movie_nstep`/`movie_intensity` so they're
plain host arrays again afterwards, same as the CPU path. Called from
[`process_slowlight_images!`](@ref) when `engine = :GPU`.

# Arguments
- `movie_nstep`: Remaining trajectory steps per pixel/frame, overwritten
  in place.
- `movie_intensity`: Accumulated intensity per pixel/frame, overwritten
  in place.
- `all_geodesics`: Matrix of pre-traced geodesic trajectories, one per
  pixel.
- `nsteps`: Matrix of trajectory lengths, one per pixel.
- `max_nstep`: Longest trajectory length across all pixels.
- `tile_height`: Number of image rows per tile (see [`gpu_tile_plan`](@ref)).
- `valid_ks`: Indices of the currently-active frames to render this
  round.
- `nimgs_concurrently`: Number of frames rendered concurrently.
- `target_times`: Target simulation time for each frame.
- `pixels_x`, `pixels_y`: Image resolution.
- `params_slowlight`: Slow-light run state (time window `[tA, tB, tf]`).
- `freq`: Frequency, in cgs units.
- `model`: Iharm model parameters.
- `gpu_data`: 3-element vector of GPU-resident GRMHD snapshots.
"""
function Slowlight.render_round_gpu!(
    movie_nstep, movie_intensity, all_geodesics, nsteps, max_nstep, tile_height, valid_ks, nimgs_concurrently,
    target_times, pixels_x, pixels_y, params_slowlight::OfSlowLight, freq, model, gpu_data
)
    threads_per_block = (16, 16)
    valid_mask_host = zeros(Int, nimgs_concurrently)
    for k in valid_ks
        valid_mask_host[k] = 1
    end
    d_valid_mask = CuArray(valid_mask_host)
    d_target_times = CuArray(target_times)
    data_tuple = Tuple(gpu_data)

    n_tiles = cld(pixels_y, tile_height)
    println("Rendering $(length(valid_ks)) frame(s) this round on GPU ($n_tiles tile(s) of height $tile_height)...")

    traj_tile_host = Array{OfTrajGeneric{Float64}}(undef, pixels_x, tile_height, max_nstep)
    for j0 in 1:tile_height:pixels_y
        j1 = min(j0 + tile_height - 1, pixels_y)
        tile_ny = j1 - j0 + 1

        traj_view = tile_ny == tile_height ? traj_tile_host : Array{OfTrajGeneric{Float64}}(undef, pixels_x, tile_ny, max_nstep)
        pack_trajectory_tile!(traj_view, all_geodesics, nsteps, pixels_x, j0, j1)
        d_traj = CuArray(traj_view)

        d_nstep = CuArray(@view movie_nstep[:, j0:j1, :])
        d_intensity = CuArray(@view movie_intensity[:, j0:j1, :])

        blocks_per_grid = (cld(pixels_x, threads_per_block[1]), cld(tile_ny, threads_per_block[2]))
        @cuda threads = threads_per_block blocks = blocks_per_grid slowlight_kernel!(
            d_traj, d_nstep, d_intensity, d_valid_mask, d_target_times,
            params_slowlight.tA, params_slowlight.tB, params_slowlight.tf,
            freq, model.a, model, data_tuple
        )
        CUDA.synchronize()

        copyto!(@view(movie_nstep[:, j0:j1, :]), Array(d_nstep))
        copyto!(@view(movie_intensity[:, j0:j1, :]), Array(d_intensity))
    end
    return nothing
end


"""
    slowlight_kernel!(traj, nstep_state, intensity, valid_mask, target_times,
        tA, tB, tf, freq, bhspin, model, data)

The GPU kernel itself: one thread per pixel `(i, j)`, looping over every
active frame `k` (skipping any where `valid_mask[k] == 0`) and continuing
that pixel's intensity integration along its trajectory. Same math as
[`render_round_cpu!`](@ref)'s inner loop, just run on the GPU. Launched
via `@cuda` from [`render_round_gpu!`](@ref).

# Arguments
- `traj`: Packed per-pixel geodesic trajectories for this tile (see
  [`pack_trajectory_tile!`](@ref)).
- `nstep_state`: Remaining trajectory steps per pixel/frame, overwritten
  in place.
- `intensity`: Accumulated intensity per pixel/frame, overwritten in
  place.
- `valid_mask`: Which frames are currently active (nonzero entries).
- `target_times`: Target simulation time for each frame.
- `tA`, `tB`: Time window currently bracketed by `data`.
- `tf`: Simulation time of the newest available dump.
- `freq`: Frequency, in cgs units.
- `bhspin`: Dimensionless black hole spin parameter.
- `model`: Iharm model parameters.
- `data`: GPU-resident, 3-snapshot `NTuple` of GRMHD snapshots (see
  [`refresh_gpu_data!`](@ref)).
"""
function slowlight_kernel!(
    traj, nstep_state, intensity, valid_mask, target_times,
    tA::Float64, tB::Float64, tf::Float64,
    freq::Float64, bhspin::Float64, model, data
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    nx, ny, _ = size(traj)
    nk = length(valid_mask)

    if i <= nx && j <= ny
        @inbounds for k in 1:nk
            if valid_mask[k] == 0
                continue
            end

            nstep = nstep_state[i, j, k]
            Intensity = intensity[i, j, k]
            dt = target_times[k] + 1e-5

            while nstep > 2
                Xi = traj[i, j, nstep].X
                Xf = traj[i, j, nstep-1].X
                Kconi = traj[i, j, nstep].Kcon
                Kconf = traj[i, j, nstep-1].Kcon

                Xi = SVector{4,Float64}(Xi[1] + dt, Xi[2], Xi[3], Xi[4])
                Xf = SVector{4,Float64}(Xf[1] + dt, Xf[2], Xf[3], Xf[4])
                if Xi[1] < tA
                    shift = tA - Xi[1]
                    Xf = SVector{4,Float64}(Xf[1] + shift, Xf[2], Xf[3], Xf[4])
                    Xi = SVector{4,Float64}(tA, Xi[2], Xi[3], Xi[4])
                end
                if Xi[1] >= tB
                    if Xf[1] >= tf
                        shift = tf - Xf[1]
                        Xi = SVector{4,Float64}(Xi[1] + shift, Xi[2], Xi[3], Xi[4])
                        Xf = SVector{4,Float64}(tf, Xf[2], Xf[3], Xf[4])
                    else
                        break
                    end
                end

                ji, ki = Radiation.get_jk(Xi, Kconi, freq, bhspin, model, data)
                jf, kf = Radiation.get_jk(Xf, Kconf, freq, bhspin, model, data)

                Intensity = Radiation.approximate_solve(Intensity, ji, ki, jf, kf, traj[i, j, nstep-1].dl)

                nstep -= 1
            end

            nstep_state[i, j, k] = nstep
            intensity[i, j, k] = Intensity
        end
    end
    return nothing
end


"""
    gpu_tile_plan(pixels_x, pixels_y, max_nstep, nimgs_concurrently, simulation_data)

Decide how many image rows fit in one GPU tile, by checking how much
memory is actually free right now -- on both the GPU (`CUDA.available_memory()`)
and the host (`Sys.free_memory()`, since [`pack_trajectory_tile!`](@ref)
builds each tile on the host first) -- and picking the size that fits
the tighter of the two. Called once from [`process_slowlight_images!`](@ref)
when `engine = :GPU`, before the rounds loop starts.

# Arguments
- `pixels_x`, `pixels_y`: Image resolution.
- `max_nstep`: Longest trajectory length across all pixels.
- `nimgs_concurrently`: Number of frames rendered concurrently.
- `simulation_data`: 3-element window of loaded GRMHD snapshots, used to
  estimate their GPU memory footprint.

# Returns
- `tile_height`: number of image rows (the `j` dimension) per tile.
"""
function Slowlight.gpu_tile_plan(pixels_x, pixels_y, max_nstep, nimgs_concurrently, simulation_data)
    traj_elem_bytes = sizeof(OfTrajGeneric{Float64})

    grid_bytes = 3 * Base.summarysize(simulation_data[1])
    state_bytes = pixels_x * pixels_y * nimgs_concurrently * (sizeof(Float64) + sizeof(Int))

    gpu_usable_bytes = max(CUDA.available_memory() - grid_bytes - state_bytes, 0)
    host_usable_bytes = Sys.free_memory()

    safety_fraction = 0.6 
    usable_bytes = floor(Int, min(gpu_usable_bytes, host_usable_bytes) * safety_fraction)

    row_bytes = pixels_x * max_nstep * traj_elem_bytes
    tile_height = row_bytes <= 0 ? pixels_y : clamp(usable_bytes ÷ row_bytes, 1, pixels_y)
    return tile_height
end

function Imaging.render_image_gpu!(Image, model, gpu_sim_data, ro, θo, phi, freq, fovx, fovy, nx, ny;
    nmaxstep=16000, nmaxstep_ceiling=50000, block_size=64)
    
    threads_per_block = (16, 16)
    blocks_per_grid = (cld(block_size, threads_per_block[1]), cld(block_size, threads_per_block[2]))

    T = promote_type(typeof(ro), typeof(θo), typeof(phi), typeof(model.a))

    Xcam = SVector{4, Float64}(Camera.camera_position(ro, θo, phi, model.a, model))
    _, Econ, _ = Tetrads.make_camera_tetrad(Xcam, model.a, model)
    Econ = SMatrix{4, 4, Float64, 16}(Econ)

    d_traj = CuArray{GPUTrajStep{T}}(undef, block_size, block_size, nmaxstep)
    d_truncated = CUDA.zeros(Bool, block_size, block_size)
    d_Image = CUDA.zeros(Float64, nx, ny)

    CUDA.@time begin
        for i_offset in 0:block_size:(nx - 1)
            for j_offset in 0:block_size:(ny - 1)
                # Grow nmaxstep until the tile has no truncated geodesic or the ceiling is hit.
                while true
                    CUDA.fill!(d_truncated, false)
                    @cuda threads=threads_per_block blocks=blocks_per_grid Imaging.raytrace_image_gpu!(
                        d_traj, d_Image, d_truncated,
                        i_offset, j_offset, block_size, block_size,
                        Xcam, Econ, model.a, nx, ny, nmaxstep,
                        freq, fovx, fovy, model.Rout, model.rmax_geo, gpu_sim_data, model
                    )
                    CUDA.synchronize()

                    any(d_truncated) || break

                    if nmaxstep >= nmaxstep_ceiling
                        @warn "Tile (i_offset=$i_offset, j_offset=$j_offset) still truncated at the absolute step ceiling ($nmaxstep_ceiling); keeping its result as-is."
                        break
                    end

                    nmaxstep = min(nmaxstep * 2, nmaxstep_ceiling)
                    println("Tile (i_offset=$i_offset, j_offset=$j_offset) truncated a geodesic; retrying with nmaxstep = $nmaxstep")
                    CUDA.unsafe_free!(d_traj)
                    d_traj = CuArray{GPUTrajStep{T}}(undef, block_size, block_size, nmaxstep)
                end
            end
        end
        CUDA.synchronize()
    end

    copyto!(Image, d_Image)
    CUDA.unsafe_free!(d_traj)
    return nmaxstep
end

# Polarized imaging (Stokes Q, U, V): kernel and launcher.
include("polarized_kernel.jl")

end
