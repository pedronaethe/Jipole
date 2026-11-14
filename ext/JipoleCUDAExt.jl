"""
CUDA backend for Jipole. Julia loads this module automatically once both `Jipole` and
`CUDA` are loaded, so CPU-only runs never pay for loading CUDA.jl.
"""
module JipoleCUDAExt

using CUDA
using StaticArrays
using Adapt
using Jipole
using Jipole: Constants, Camera, Geodesics, Radiation, Iharm, Imaging, Polarization, Slowlight, Utils_GPU, Tetrads
using Jipole.GeoTypes: OfTrajGeneric, GPUTrajStep
using Jipole.Slowlight: OfSlowLight, shifted_points
  """
      copy_iharm_to_gpu(cpu_data)

  Copy a snapshot to the GPU: every array (fluid fields, derived quantities and the block
  mesh) becomes a CuArray of the same element type; the time stays a number.
  """
  function Utils_GPU.copy_iharm_to_gpu(cpu_data)
      return Adapt.adapt(CuArray, cpu_data)
  end


  """
      raytrace_image_gpu!(d_traj, d_Image, d_pol, d_truncated, i_offset, j_offset, block_size_x, block_size_y,
          Xcam, Econ, Ecov, bhspin, nx, ny, freq, fovx, fovy, Rstop, qu_conv, data, params)

  GPU kernel for one tile of the image: each thread takes one pixel of the tile and runs
  [`calculate_image!`](@ref) on it. Launched once per tile by `render_image_gpu!`, because the
  trajectory buffer only holds one tile.

  Pass `nothing` as `d_pol` for an unpolarized image. Julia then compiles a separate kernel
  with no polarized code in it, so the unpolarized image costs the same as before.
  """
  function Imaging.raytrace_image_gpu!(
      d_traj, d_Image, d_pol, d_truncated,
      i_offset, j_offset, block_size_x, block_size_y,
      Xcam, Econ, Ecov, bhspin, nx, ny,
      freq, fovx, fovy, Rstop, qu_conv, data, params
  )
      local_i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
      local_j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

      i = local_i - 1 + i_offset
      j = local_j - 1 + j_offset

      if local_i <= block_size_x && local_j <= block_size_y && i < nx && j < ny
          calculate_image!(
              d_traj, d_Image, d_pol, d_truncated, Xcam, Econ, Ecov, bhspin, nx, ny,
              i, j, local_i, local_j, freq, fovx, fovy, Rstop, qu_conv, params, data
          )
      end
      return nothing
  end

"""
      calculate_image!(traj, d_Image, d_pol, d_truncated, Xcam, Econ, Ecov, bhspin, nx, ny,
          i_global, j_global, i_local, j_local, freq, fovx, fovy, Rstop, qu_conv, params, data=nothing)

  Per-pixel body of the GPU kernel. Traces the geodesic of pixel (i_global, j_global) with
  `Geodesics.trace_geodesic!` into this pixel's slice of the tile buffer, then integrates the
  transfer from the far end back to the camera and writes the pixel of `d_Image` and, when
  `d_pol` is an array, of `d_pol` (Stokes I, Q, U, V and Faraday depth).

  Contains nothing CUDA-specific, so test/gpu_pol runs it on the CPU with ordinary arrays.
  Keep it that way: no CUDA macros such as `CUDA.@cuassert` here (they cannot compile for the CPU).

  The trajectory buffer holds `GPUTrajStep`s, without the half-step point. The polarized
  transport needs it, so it is recomputed for each step from the step's near end: two extra
  connection evaluations per step, in exchange for 72 instead of 136 bytes per point.

  # Arguments
  - `traj`: Trajectory buffer of the tile, `(block_size, block_size, nmaxstep)` `GPUTrajStep`s.
  - `d_Image`: Unpolarized image, written at `(i_global + 1, j_global + 1)`.
  - `d_pol`: Polarized image of size `(NIMG, nx, ny)`, or `nothing` for an unpolarized run.
  - `d_truncated`: Set to `true` at `(i_local, j_local)` if the buffer filled up before the geodesic ended.
  - `Xcam`: Camera position in internal coordinates.
  - `Econ`, `Ecov`: Camera tetrad, built once per image.
  - `bhspin`: Dimensionless black hole spin parameter.
  - `nx`, `ny`: Full image resolution.
  - `i_global`, `j_global`: Pixel indices in the full image (0-based).
  - `i_local`, `j_local`: Pixel indices within the tile (1-based).
  - `freq`: Observing frequency, in cgs units.
  - `fovx`, `fovy`: Field of view, in radians.
  - `Rstop`: Backward-integration stopping radius.
  - `qu_conv`: Q/U sign convention, see `Polarization.save_pixel!`.
  - `params`: Model parameters.
  - `data`: Model data (e.g. `Iharm`'s GRMHD snapshots).
  """
function calculate_image!(
      traj, d_Image, d_pol, d_truncated,
      Xcam::SVector{4,Float64}, Econ::SMatrix{4,4,Float64}, Ecov::SMatrix{4,4,Float64}, bhspin::Float64,
      nx::Int64, ny::Int64,
      i_global::Int64, j_global::Int64,
      i_local::Int64, j_local::Int64,
      freq::Float64, fovx::Float64, fovy::Float64, Rstop::Float64, qu_conv::Int64, params,
      data::T_data = nothing
  ) where {T_data}

    if (i_global >= nx || j_global >= ny)
        return nothing
    end
    # Known when the kernel is compiled: with d_pol = nothing the polarized code is not compiled in.
    polarized = d_pol !== nothing

    Kcon0 = Geodesics.init_kcon(i_global, j_global, Econ, nx, ny, fovx, fovy)
    Kcon = Kcon0 * (freq * Constants.HPL / (Constants.ME * Constants.CL * Constants.CL))
    dl_unit::Float64 = params.L_unit * Constants.HPL / (Constants.ME * Constants.CL^2)
    Rh = 1.0 + sqrt(1.0 - bhspin * bhspin)

    # This pixel's slice of the tile buffer; the tracer stops when it is full.
    ptraj = view(traj, i_local, j_local, :)
    step, truncated = Geodesics.trace_geodesic!(ptraj, Xcam, Kcon, bhspin, Rh, Rstop, params)
    @inbounds d_truncated[i_local, j_local] = truncated

    # Radiative transfer, from the far end of the geodesic to the camera.
    Intensity = 0.0
    N = Polarization.zero_tensor(Float64)
    tauF = 0.0

    @inbounds far = ptraj[step]
    ji, ki = Radiation.get_jk(far.X, far.Kcon, freq, bhspin, params, data)

    @inbounds for nstep in step:-1:2
        ti = ptraj[nstep]
        tf = ptraj[nstep - 1]
        if polarized
            # Half-step point of this step, recomputed from its near end.
            _, _, Xhalf, Kconhalf = Geodesics.push_photon(tf.X, tf.Kcon, -tf.dl / dl_unit, bhspin, params)
            # Parallel transport, then sources if inside the radiating region.
            N, tauF = Polarization.polarized_step(N, tauF, ti.X, ti.Kcon, Xhalf, Kconhalf, tf.X, tf.Kcon,
                tf.dl, dl_unit, Rh, freq, bhspin, params, data)
        end

        if Radiation.radiating_region(tf.X, params, Rh)
            jf, kf = Radiation.get_jk(tf.X, tf.Kcon, freq, bhspin, params, data)
            Intensity = Radiation.approximate_solve(Intensity, ji, ki, jf, kf, tf.dl)
            ji = jf
            ki = kf
        end
    end

    @inbounds d_Image[i_global + 1, j_global + 1] = Intensity * (freq^3)
    if polarized
        SI, SQ, SU, SV = Polarization.project_n(N, Ecov)
        Polarization.save_pixel!(d_pol, i_global + 1, j_global + 1, SI, SQ, SU, SV, tauF, freq, qu_conv)
    end
    return nothing
end

"""
    render_round_gpu!(movie_nstep, movie_intensity, movie_N, movie_tauF, rays, offsets, tile_height,
        valid_ks, target_times, params_slowlight, freq, model, gpu_data)

Render one round on the GPU -- the `:GPU` counterpart of [`render_round_cpu!`](@ref).
The rays are stored back to back at their own lengths (see [`flatten_rays`](@ref)).
When they all fit on the GPU, `rays` is already a GPU array, uploaded once for the
whole run; otherwise it is the host array and each tile (a block of image rows, sized
by [`gpu_tile_plan`](@ref)) uploads its own contiguous slice. For each tile: upload
the per-frame state (`movie_nstep`/`movie_intensity`, and `movie_N`/`movie_tauF` in a
polarized run), run [`slowlight_kernel!`](@ref) with one layer of 8x8-thread blocks per
open frame, then copy the state back so it is plain host arrays again afterwards, same
as the CPU path. Called from [`process_slowlight_images!`](@ref) when `engine = :GPU`.

# Arguments
- `movie_nstep`: Remaining trajectory steps per pixel/frame, overwritten in place.
- `movie_intensity`: Accumulated intensity per pixel/frame, overwritten in place.
- `movie_N`, `movie_tauF`: Coherency tensor and Faraday depth per pixel/frame,
overwritten in place, or `nothing` for an unpolarized run.
- `rays`: All rays, back to back (a GPU array when they all fit, else the host array).
- `offsets`: Where each pixel's ray starts in `rays` (its point `s` is `rays[offsets[i, j] + s]`).
- `tile_height`: Number of image rows per tile (see [`gpu_tile_plan`](@ref)).
- `valid_ks`: Indices of the currently-active frames to render this round.
- `target_times`: Target simulation time for each frame.
- `params_slowlight`: Slow-light run state (time window `[tA, tB, tf]`).
- `freq`: Frequency, in cgs units.
- `model`: Iharm model parameters.
- `gpu_data`: 3-element vector of GPU-resident GRMHD snapshots.
"""
function Slowlight.render_round_gpu!(
    movie_nstep, movie_intensity, movie_N, movie_tauF, rays, offsets, tile_height, valid_ks,
    target_times, params_slowlight::OfSlowLight, freq, model, gpu_data
)
    # 8x8 blocks: a block keeps its place on the GPU until its slowest thread is done, and small
    # square blocks hold neighbouring pixels with similar ray lengths (1.4x faster than 16x16).
    threads_per_block = (8, 8)
    pixels_x, pixels_y = size(offsets)
    n_tiles = cld(pixels_y, tile_height)
    println("Rendering $(length(valid_ks)) frame(s) this round on GPU ($n_tiles tile(s) of height $tile_height)...")
    isempty(valid_ks) && return nothing

    d_valid_ks = CuArray(valid_ks)
    d_target_times = CuArray(target_times)
    data_tuple = Tuple(gpu_data)

    for j0 in 1:tile_height:pixels_y
        j1 = min(j0 + tile_height - 1, pixels_y)
        tile_ny = j1 - j0 + 1

        # The tile's rays: all of them if already on the GPU, else the tile's contiguous slice.
        if rays isa CuArray
            d_rays = rays
            d_offsets = CuArray(offsets[:, j0:j1])
        else
            first = offsets[1, j0] + 1
            last = j1 < pixels_y ? offsets[1, j1 + 1] : length(rays)
            d_rays = CuArray{eltype(rays)}(undef, last - first + 1)
            copyto!(d_rays, 1, rays, first, last - first + 1)
            d_offsets = CuArray(offsets[:, j0:j1] .- (first - 1))
        end
        d_nstep = CuArray(@view movie_nstep[:, j0:j1, :])
        d_intensity = CuArray(@view movie_intensity[:, j0:j1, :])
        d_N = movie_N === nothing ? nothing : CuArray(@view movie_N[:, j0:j1, :])
        d_tauF = movie_tauF === nothing ? nothing : CuArray(@view movie_tauF[:, j0:j1, :])

        # one layer of blocks per open frame: every frame of every pixel runs in parallel
        blocks_per_grid = (cld(pixels_x, threads_per_block[1]), cld(tile_ny, threads_per_block[2]),
        length(valid_ks))
        @cuda threads = threads_per_block blocks = blocks_per_grid slowlight_kernel!(
            d_rays, d_offsets, d_nstep, d_intensity, d_N, d_tauF, d_valid_ks, d_target_times,
            params_slowlight.tA, params_slowlight.tB, params_slowlight.tf,
            freq, model.a, model, data_tuple
        )
        CUDA.synchronize()

        copyto!(@view(movie_nstep[:, j0:j1, :]), Array(d_nstep))
        copyto!(@view(movie_intensity[:, j0:j1, :]), Array(d_intensity))
        if movie_N !== nothing
            copyto!(@view(movie_N[:, j0:j1, :]), Array(d_N))
            copyto!(@view(movie_tauF[:, j0:j1, :]), Array(d_tauF))
        end
    end
    return nothing
end


"""
    slowlight_kernel!(rays, offsets, nstep_state, intensity, N_state, tauF_state, valid_ks,
target_times,
        tA, tB, tf, freq, bhspin, model, data)

The GPU kernel itself: one thread per pixel `(i, j)` and open frame (`valid_ks[blockIdx().z]`),
so all frames of all pixels run in parallel. Each thread continues that frame's integration
along the pixel's ray (`rays[offsets[i, j] + 1]` onward), with the same math as
[`render_round_cpu!`](@ref)'s inner loop, in two passes over the same steps:

1. the intensity, up to the step where the frame leaves the current data window;
2. in a polarized run, the coherency tensor and Faraday depth over those same steps.

Two passes keep the coherency tensor (64 registers) out of the intensity's emissivity
calls, which cuts the polarized kernel's spills and run time by about 15%. With
`N_state = nothing` the second pass is not compiled into the kernel. Launched via `@cuda`
from [`render_round_gpu!`](@ref).

# Arguments
- `rays`: The tile's rays, back to back (see [`flatten_rays`](@ref)).
- `offsets`: Where each pixel's ray starts in `rays`.
- `nstep_state`: Remaining trajectory steps per pixel/frame, overwritten in place.
- `intensity`: Accumulated intensity per pixel/frame, overwritten in place.
- `N_state`, `tauF_state`: Coherency tensor and Faraday depth per pixel/frame,
overwritten in place, or `nothing` for an unpolarized run.
- `valid_ks`: Indices of the open frames; block layer `z` works on frame `valid_ks[z]`.
- `target_times`: Target simulation time for each frame.
- `tA`, `tB`: Time window currently bracketed by `data`.
- `tf`: Simulation time of the newest available dump.
- `freq`: Frequency, in cgs units.
- `bhspin`: Dimensionless black hole spin parameter.
- `model`: Iharm model parameters.
- `data`: GPU-resident, 3-snapshot `NTuple` of GRMHD snapshots (kept in step
with the CPU window by `process_slowlight_images!`).
"""
 function slowlight_kernel!(
    rays, offsets, nstep_state, intensity, N_state, tauF_state, valid_ks, target_times,
    tA::Float64, tB::Float64, tf::Float64,
    freq::Float64, bhspin::Float64, model, data
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    nx, ny = size(offsets)
    (i <= nx && j <= ny) || return nothing

    @inbounds k = valid_ks[blockIdx().z]
    @inbounds off = offsets[i, j]

    @inbounds begin
        nstep_start = nstep_state[i, j, k]
        dt = target_times[k] + 1e-5

        # Pass 1: the intensity.
        nstep = nstep_start
        Intensity = intensity[i, j, k]
        while nstep > 2
            Xi, _, Xf, stop = shifted_points(rays, off, nstep, dt, tA, tB, tf)
            stop && break
            ji, ki = Radiation.get_jk(Xi, rays[off + nstep].Kcon, freq, bhspin, model, data)
            jf, kf = Radiation.get_jk(Xf, rays[off + nstep - 1].Kcon, freq, bhspin, model, data)
            Intensity = Radiation.approximate_solve(Intensity, ji, ki, jf, kf, rays[off + nstep - 1].dl)
            nstep -= 1
        end
        nstep_state[i, j, k] = nstep
        intensity[i, j, k] = Intensity

        # Pass 2 (polarized run): the coherency tensor and Faraday depth, over the same steps.
        if N_state !== nothing
            Rh = 1.0 + sqrt(1.0 - bhspin * bhspin)
            dl_unit = model.L_unit * Constants.HPL / (Constants.ME * Constants.CL * Constants.CL)
            N = N_state[i, j, k]
            tauF = tauF_state[i, j, k]
            for s in nstep_start:-1:(nstep + 1)
                Xi, Xh, Xf, _ = shifted_points(rays, off, s, dt, tA, tB, tf)
                N, tauF = Polarization.polarized_step(N, tauF, Xi, rays[off + s].Kcon, Xh, rays[off + s].Kconhalf,
                Xf, rays[off + s - 1].Kcon, rays[off + s - 1].dl, dl_unit, Rh, freq, bhspin,
                model, data)
            end
            N_state[i, j, k] = N
            tauF_state[i, j, k] = tauF
        end
    end
    return nothing
end

"""
    gpu_tile_plan(nsteps, nimgs_concurrently, simulation_data; pol=false)

Decide how many image rows fit in one GPU tile, from the GPU memory free right now
(`CUDA.available_memory()`) and the rays' real lengths (`nsteps`). When the answer is
the whole image, [`process_slowlight_images!`](@ref) uploads all rays once and keeps
them. Called once from [`process_slowlight_images!`](@ref) when `engine = :GPU`,
before the rounds loop starts.

# Arguments
- `nsteps`: Matrix of trajectory lengths, one per pixel.
- `nimgs_concurrently`: Number of frames rendered concurrently.
- `simulation_data`: 3-element window of loaded GRMHD snapshots, used to
estimate their GPU memory footprint.
- `pol`: Whether the per-frame polarized state is kept too.

# Returns
- `tile_height`: number of image rows (the `j` dimension) per tile.
"""
function Slowlight.gpu_tile_plan(nsteps, nimgs_concurrently, simulation_data; pol::Bool = false)
    pixels_x, pixels_y = size(nsteps)
    grid_bytes = 3 * Base.summarysize(simulation_data[1])
    # nstep and intensity per pixel/frame, plus the coherency tensor and Faraday depth when polarized
    state_elem_bytes = sizeof(Float64) + sizeof(Int) + (pol ? sizeof(SMatrix{4,4,ComplexF64,16}) +
sizeof(Float64) : 0)
    state_bytes = pixels_x * pixels_y * nimgs_concurrently * state_elem_bytes

    safety_fraction = 0.6
    usable_bytes = floor(Int, max(CUDA.available_memory() - grid_bytes - state_bytes, 0) *
safety_fraction)

    # bytes of the image row with the most ray points
    row_bytes = maximum(sum(nsteps; dims = 1)) * sizeof(OfTrajGeneric{Float64})
    return clamp(usable_bytes ÷ row_bytes, 1, pixels_y)
end

function Imaging.render_image_gpu!(Image, model, gpu_sim_data, ro, θo, phi, freq, fovx, fovy, nx, ny;
      pol=nothing, qu_conv::Int=0, nmaxstep=16000, nmaxstep_ceiling=50000, block_size=64)

      threads_per_block = (16, 16)
      blocks_per_grid = (cld(block_size, threads_per_block[1]), cld(block_size, threads_per_block[2]))

      T = promote_type(typeof(ro), typeof(θo), typeof(phi), typeof(model.a))

      Xcam = SVector{4, Float64}(Camera.camera_position(ro, θo, phi, model.a, model))
      _, Econ, Ecov = Tetrads.make_camera_tetrad(Xcam, model.a, model)
      Econ = SMatrix{4, 4, Float64, 16}(Econ)
      Ecov = SMatrix{4, 4, Float64, 16}(Ecov)

      d_traj = CuArray{GPUTrajStep{T}}(undef, block_size, block_size, nmaxstep)
      d_truncated = CUDA.zeros(Bool, block_size, block_size)
      d_Image = CUDA.zeros(Float64, nx, ny)
      d_pol = pol === nothing ? nothing : CUDA.zeros(Float64, Polarization.NIMG, nx, ny)

      CUDA.@time begin
        for i_offset in 0:block_size:(nx - 1)
            for j_offset in 0:block_size:(ny - 1)
                # Grow nmaxstep until the tile has no truncated geodesic or the ceiling is hit.
                while true
                    CUDA.fill!(d_truncated, false)
                    @cuda threads=threads_per_block blocks=blocks_per_grid Imaging.raytrace_image_gpu!(
                        d_traj, d_Image, d_pol, d_truncated,
                        i_offset, j_offset, block_size, block_size,
                        Xcam, Econ, Ecov, model.a, nx, ny,
                        freq, fovx, fovy, model.rmax_geo, qu_conv, gpu_sim_data, model
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
    pol === nothing || copyto!(pol, d_pol)
    CUDA.unsafe_free!(d_traj)
    all(isfinite, Image) || error("NaN or Inf intensity in the GPU image.")
    return nmaxstep
end

end
