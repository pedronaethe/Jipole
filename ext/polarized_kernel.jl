# Polarized imaging on the GPU. Included by JipoleCUDAExt.jl, inside its module.
#
# STATUS: this kernel has not been run on a GPU. Its per-pixel body, `calculate_image_pol!`,
# contains nothing CUDA-specific and has been run on the CPU with ordinary arrays, where it
# reproduces the CPU polarized image (test/gpu_pol). What remains untested is everything a GPU
# adds: compiling the kernel with CUDA.jl, and launching it.
#
# The physics is not repeated here: every geodesic step calls
# `Polarization.polarized_step`, the same routine the CPU loop
# (`Polarization.integrate_emission_pol`) calls.

using Jipole: Polarization, ImagingPol

"""
    trace_geodesic_gpu!(traj, i_local, j_local, X, K, dl_unit, Rh, Rstop, nmaxstep, bhspin, params)

Trace the geodesic of one pixel backward from the camera, storing each point
into `traj[i_local, j_local, :]`: the trace stage of [`calculate_image!`](@ref),
as a function. The storage convention is the one of that kernel: the step
length is stored with the *far* point of the step, and no half-step values are
kept.

# Arguments
- `traj`: Scratch buffer of `GPUTrajStep`s for the tile.
- `i_local`, `j_local`: Pixel indices within the tile.
- `X`, `K`: Camera position and photon 4-momentum at the camera.
- `dl_unit`: Conversion `L_unit · h/(m_e c²)` applied to the stored step lengths.
- `Rh`, `Rstop`: Event horizon radius and backward-integration stopping radius.
- `nmaxstep`: Maximum number of points (the third dimension of `traj`).
- `bhspin`: Dimensionless black hole spin parameter.
- `params`: Model parameters.

# Returns
- `(step, truncated)`: the number of points stored, and whether `nmaxstep` was
  reached before the geodesic met its stopping condition.
"""
@inline function trace_geodesic_gpu!(traj, i_local, j_local, X, K, dl_unit, Rh, Rstop, nmaxstep, bhspin, params)
    T = eltype(X)
    step = 1
    @inbounds traj[i_local, j_local, step] = GPUTrajStep{T}(zero(T), X, K)
    while (Geodesics.stop_backward_integration(X, K, Rh, Rstop) == 0 && (step < nmaxstep))
        dl = Geodesics.stepsize(X, K, params.cstartx, params.cstopx)
        X, K, _, _ = Geodesics.push_photon(X, K, -dl, bhspin, params)
        step += 1
        @inbounds traj[i_local, j_local, step] = GPUTrajStep{T}(dl * dl_unit, X, K)
    end
    truncated = (step >= nmaxstep) && (Geodesics.stop_backward_integration(X, K, Rh, Rstop) == 0)
    return step, truncated
end

"""
    calculate_image_pol!(traj, d_Image, d_pol, d_truncated, Xcam, Econ, Ecov, bhspin, nx, ny, nmaxstep,
        i_global, j_global, i_local, j_local, freq, fovx, fovy, Rstop, qu_conv, params, data)

Per-pixel body of the polarized GPU kernel: the polarized counterpart of
[`calculate_image!`](@ref). Traces the geodesic of pixel `(i_global, j_global)`
backward from the camera, then walks it forward once, integrating both the
unpolarized intensity (as `calculate_image!` does) and the coherency tensor
(`Polarization.polarized_step`), and stores `d_Image[i, j]` and
`d_pol[:, i, j]`.

The trajectory buffer holds `GPUTrajStep`s, which have no half-step point. The
parallel transport needs it, so it is recomputed for each step by repeating
that step of the geodesic integrator from the near point. This costs two
connection evaluations per step and keeps the buffer at 72 bytes per point; the
alternative is to store `OfTrajGeneric`s (136 bytes per point).

# Arguments
- `traj`: Scratch buffer of `GPUTrajStep`s for the tile.
- `d_Image`: Unpolarized image, overwritten at `(i_global + 1, j_global + 1)`.
- `d_pol`: Polarized image of size `(NIMG, nx, ny)`, overwritten for the same pixel.
- `d_truncated`: Set `true` at `(i_local, j_local)` if `nmaxstep` was too small.
- `Xcam`: Camera position in internal coordinates.
- `Econ`, `Ecov`: Camera tetrad (`Tetrads.make_camera_tetrad`), built once per image.
- `bhspin`: Dimensionless black hole spin parameter.
- `nx`, `ny`: Full image resolution.
- `nmaxstep`: Maximum number of geodesic integration steps.
- `i_global`, `j_global`: Pixel indices in the full image plane (0-based).
- `i_local`, `j_local`: Pixel indices within this tile's `traj` buffer.
- `freq`: Observing frequency, in cgs units.
- `fovx`, `fovy`: Field of view, in radians.
- `Rstop`: Backward-integration stopping radius.
- `qu_conv`: Q/U sign convention, see `Polarization.save_pixel!`.
- `params`: Model parameters.
- `data`: Model-specific auxiliary data (e.g. `Iharm`'s GRMHD snapshots).
"""
function calculate_image_pol!(
    traj, d_Image, d_pol, d_truncated,
    Xcam::SVector{4,Float64}, Econ::SMatrix{4,4,Float64}, Ecov::SMatrix{4,4,Float64}, bhspin::Float64,
    nx::Int64, ny::Int64, nmaxstep::Int64,
    i_global::Int64, j_global::Int64,
    i_local::Int64, j_local::Int64,
    freq::Float64, fovx::Float64, fovy::Float64, Rstop::Float64, qu_conv::Int64, params,
    data::T_data = nothing
) where {T_data}

    if (i_global >= nx || j_global >= ny)
        return nothing
    end
    Kcon0 = Geodesics.init_kcon(i_global, j_global, Econ, nx, ny, fovx, fovy)
    Kcon = Kcon0 * (freq * Constants.HPL / (Constants.ME * Constants.CL * Constants.CL))

    dl_unit::Float64 = params.L_unit * Constants.HPL / (Constants.ME * Constants.CL^2)
    Rh = 1.0 + sqrt(1.0 - bhspin * bhspin)

    step, truncated = trace_geodesic_gpu!(traj, i_local, j_local, Xcam, Kcon, dl_unit, Rh, Rstop, nmaxstep, bhspin, params)
    @inbounds d_truncated[i_local, j_local] = truncated

    # Radiative transfer, from the far end of the geodesic to the camera.
    Intensity = 0.0
    N = Polarization.zero_tensor(Float64)
    tauF = 0.0

    @inbounds far = traj[i_local, j_local, step]
    ji, ki = Radiation.get_jk(far.X, far.Kcon, freq, bhspin, params, data)

    @inbounds for nstep in step:-1:2
        ti = traj[i_local, j_local, nstep]
        tf = traj[i_local, j_local, nstep-1]
        dlam = ti.dl

        # Half-step point of this step, as the geodesic integrator computed it.
        _, _, Xhalf, Kconhalf = Geodesics.push_photon(tf.X, tf.Kcon, -dlam / dl_unit, bhspin, params)

        # Polarized: parallel transport, then sources if inside the radiating region.
        N, tauF = Polarization.polarized_step(N, tauF, ti.X, ti.Kcon, Xhalf, Kconhalf, tf.X, tf.Kcon,
            dlam, dl_unit, Rh, freq, bhspin, params, data)

        # Unpolarized: same update as in calculate_image!.
        if Radiation.radiating_region(tf.X, params, Rh)
            jf, kf = Radiation.get_jk(tf.X, tf.Kcon, freq, bhspin, params, data)
            Intensity = Radiation.approximate_solve(Intensity, ji, ki, jf, kf, dlam)
            ji = jf
            ki = kf
        end
    end

    SI, SQ, SU, SV = Polarization.project_n(N, Ecov)

    @inbounds d_Image[i_global+1, j_global+1] = Intensity * (freq^3)
    Polarization.save_pixel!(d_pol, i_global + 1, j_global + 1, SI, SQ, SU, SV, tauF, freq, qu_conv)

    return nothing
end

"""
    raytrace_image_gpu_pol!(d_traj, d_Image, d_pol, d_truncated, i_offset, j_offset, block_size_x, block_size_y,
        Xcam, Econ, Ecov, bhspin, nx, ny, nmaxstep, freq, fovx, fovy, Rstop, qu_conv, data, params)

GPU kernel for one tile of the polarized image: the polarized counterpart of
`Imaging.raytrace_image_gpu!`. Maps the CUDA thread to a pixel of the tile and
calls [`calculate_image_pol!`](@ref) for it.
"""
function raytrace_image_gpu_pol!(
    d_traj, d_Image, d_pol, d_truncated,
    i_offset, j_offset, block_size_x, block_size_y,
    Xcam, Econ, Ecov, bhspin, nx, ny, nmaxstep,
    freq, fovx, fovy, Rstop, qu_conv, data, params
)
    local_i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    local_j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    i = local_i - 1 + i_offset
    j = local_j - 1 + j_offset

    if local_i <= block_size_x && local_j <= block_size_y && i < nx && j < ny
        calculate_image_pol!(
            d_traj, d_Image, d_pol, d_truncated, Xcam, Econ, Ecov, bhspin, nx, ny, nmaxstep,
            i, j, local_i, local_j, freq, fovx, fovy, Rstop, qu_conv, params, data
        )
    end
    return nothing
end

"""
    ImagingPol.render_image_gpu_pol!(Image, pol, model, gpu_sim_data, ro, θo, phi, freq, fovx, fovy, nx, ny;
        nmaxstep=16000, nmaxstep_ceiling=50000, block_size=64, qu_conv=0)

Render the unpolarized image `Image` and the polarized image `pol` (size
`(NIMG, nx, ny)`) on the GPU, tile by tile: the polarized counterpart of
`Imaging.render_image_gpu!`, with the same tiling and the same retry when a
tile's geodesics do not fit in `nmaxstep` points.

# Returns
- The `nmaxstep` the run ended on, to start the next image from.
"""
function ImagingPol.render_image_gpu_pol!(Image, pol, model, gpu_sim_data, ro, θo, phi, freq, fovx, fovy, nx, ny;
    nmaxstep=16000, nmaxstep_ceiling=50000, block_size=64, qu_conv::Int=0)

    threads_per_block = (16, 16)
    blocks_per_grid = (cld(block_size, threads_per_block[1]), cld(block_size, threads_per_block[2]))

    T = promote_type(typeof(ro), typeof(θo), typeof(phi), typeof(model.a))

    Xcam = SVector{4,Float64}(Camera.camera_position(ro, θo, phi, model.a, model))
    _, Econ, Ecov = Tetrads.make_camera_tetrad(Xcam, model.a, model)
    Econ = SMatrix{4,4,Float64,16}(Econ)
    Ecov = SMatrix{4,4,Float64,16}(Ecov)

    d_traj = CuArray{GPUTrajStep{T}}(undef, block_size, block_size, nmaxstep)
    d_truncated = CUDA.zeros(Bool, block_size, block_size)
    d_Image = CUDA.zeros(Float64, nx, ny)
    d_pol = CUDA.zeros(Float64, Polarization.NIMG, nx, ny)

    CUDA.@time begin
        for i_offset in 0:block_size:(nx - 1)
            for j_offset in 0:block_size:(ny - 1)
                # Grow nmaxstep until the tile has no truncated geodesic or the ceiling is hit.
                while true
                    CUDA.fill!(d_truncated, false)
                    @cuda threads=threads_per_block blocks=blocks_per_grid raytrace_image_gpu_pol!(
                        d_traj, d_Image, d_pol, d_truncated,
                        i_offset, j_offset, block_size, block_size,
                        Xcam, Econ, Ecov, model.a, nx, ny, nmaxstep,
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
    copyto!(pol, d_pol)
    CUDA.unsafe_free!(d_traj)
    return nmaxstep
end
