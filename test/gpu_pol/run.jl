# Polarized GPU kernel, checked without a GPU.
#
# The per-pixel body of the polarized GPU kernel (calculate_image_pol! in
# ext/polarized_kernel.jl) contains nothing CUDA-specific: it reads and writes arrays and
# calls the same routines as the CPU path. It is run here on the CPU, with ordinary arrays in
# place of CUDA device arrays, and must reproduce the image of the CPU path
# (ImagingPol.raytrace_image_pol). That covers what is specific to the kernel: its trajectory
# layout (step length stored with the far point, no half-step values), the recomputation of
# the half-step point, and the unpolarized update it carries alongside.
#
# Then the whole kernel (raytrace_image_gpu_pol!) is compiled for an NVIDIA target with
# GPUCompiler, the compiler CUDA.jl uses, and its validation is run on the result: it reports
# anything a GPU kernel cannot contain (dynamic dispatch, heap allocation, calls into the Julia
# runtime). No GPU is needed for that, only for linking and running, which are skipped. The
# existing unpolarized kernel, which is known to run on GPUs, goes through the same check as a
# control.
#
# CUDA.jl loads on a machine without a GPU. What this test cannot cover is linking the kernel
# against NVIDIA's libraries, launching it, and its results on a GPU.
#
# Usage: julia --project=scripts run.jl
using Jipole
using CUDA
using StaticArrays

const C = Jipole.Constants
const ext = Base.get_extension(Jipole, :JipoleCUDAExt)
const REPO = normpath(joinpath(@__DIR__, "..", ".."))

failures = String[]
report(ok, what) = (println(ok ? "PASS  " : "FAIL  ", what); ok || push!(failures, what); ok)
nmse(a, ref) = sum(abs2, a .- ref) / sum(abs2, ref)

dump = joinpath(REPO, "test", "data", "tiny_dump.h5")
MBH = 6.2e9; Rhigh = 20.0
ro = 1000.0; th = 163.0; phi = 0.0; freq = 230.0e9; nx = 24; nmaxstep = 8000
model, data = redirect_stdout(devnull) do
    m = Jipole.Iharm.read_header(dump, MBH; th_beg=1.74e-2, Rlow=1.0, Rhigh=Rhigh, beta_crit=1.0,
        sigma_cut=1.0, sigma_cut_high=-1.0, M_unit=C.M_UNIT_SANE)
    m, [Jipole.Iharm.load_data(dump, Rhigh, m)]
end
Rh = 1 + sqrt(1.0 - model.a^2)
fov = 16.9e6 * C.PC / model.L_unit / C.MUAS_PER_RAD * 160.0 / ro

# Reference: the CPU path.
unpol_cpu, pol_cpu = redirect_stdout(devnull) do
    Jipole.ImagingPol.raytrace_image_pol(model, data, ro, th, phi, freq, nx, nx, fov, fov, nmaxstep, Rh, 0.0, 0.0)
end

# The kernel body, pixel by pixel, on CPU arrays laid out as the GPU buffers are.
# The model data goes in as a 1-tuple, as on the GPU.
traj = Array{Jipole.GeoTypes.GPUTrajStep{Float64}}(undef, nx, nx, nmaxstep)
unpol_k = zeros(nx, nx)
pol_k = zeros(Jipole.Polarization.NIMG, nx, nx)
truncated = zeros(Bool, nx, nx)
gpu_like_data = (data[1],)
function run_kernel_body!(traj, unpol_k, pol_k, truncated, model, gpu_like_data, ro, th, phi, freq, fov, nx, nmaxstep)
    # Camera position and tetrad, built once per image as render_image_gpu_pol! does.
    Xcam = SVector{4,Float64}(Jipole.Camera.camera_position(ro, th, phi, model.a, model))
    _, Econ, Ecov = Jipole.Tetrads.make_camera_tetrad(Xcam, model.a, model)
    Econ = SMatrix{4,4,Float64,16}(Econ)
    Ecov = SMatrix{4,4,Float64,16}(Ecov)
    for i in 0:nx-1, j in 0:nx-1
        ext.calculate_image_pol!(traj, unpol_k, pol_k, truncated, Xcam, Econ, Ecov, model.a, nx, nx, nmaxstep,
            i, j, i + 1, j + 1, freq, fov, fov, model.rmax_geo, 0, model, gpu_like_data)
    end
end
run_kernel_body!(traj, unpol_k, pol_k, truncated, model, gpu_like_data, ro, th, phi, freq, fov, nx, nmaxstep)

report(!any(truncated), "no geodesic truncated at $nmaxstep points")
report(sum(pol_cpu[1, :, :]) > 0 && maximum(pol_cpu[5, :, :]) > 0, "the test image has emission and Faraday rotation")
# The CPU path reads the half-step point from the trajectory; the kernel recomputes it from a
# step length that went through one multiplication and one division. Agreement is to round-off.
report(nmse(unpol_k, unpol_cpu) < 1e-24, "unpolarized image: kernel body against CPU path, NMSE $(nmse(unpol_k, unpol_cpu))")
for (s, name) in enumerate(("Stokes I", "Stokes Q", "Stokes U", "Stokes V", "Faraday depth"))
    e = nmse(pol_k[s, :, :], pol_cpu[s, :, :])
    report(e < 1e-20, "$name: kernel body against CPU path, NMSE $e")
end

# A GPU kernel cannot allocate, and its arguments must be isbits (the device arrays stand in
# for the ordinary arrays used here).
bytes = @allocated run_kernel_body!(traj, unpol_k, pol_k, truncated, model, gpu_like_data, ro, th, phi, freq, fov, nx, nmaxstep)
report(bytes == 0, "kernel body allocates $bytes bytes over $(nx * nx) pixels")
report(isbits(model) && isbitstype(eltype(traj)), "model parameters and trajectory points are isbits")

# ---------------------------------------------------------------------------------------------
# Compile the kernels for an NVIDIA target and validate the result
# ---------------------------------------------------------------------------------------------
const GPUCompiler = CUDA.GPUCompiler
DeviceArray(T, N) = CUDA.CuDeviceArray{T,N,CUDA.AS.Global}
const DeviceData = Jipole.Iharm.IharmData{Float64,DeviceArray(Float64, 3),Float64,DeviceArray(Float64, 3)}
const TrajArray = DeviceArray(Jipole.GeoTypes.GPUTrajStep{Float64}, 3)
# Declared but undefined only because nothing is linked here: GPUCompiler's routines for
# reporting an exception raised inside a kernel. NVIDIA's math library (__nv_*) is the other.
const EXCEPTION_RUNTIME = ("gpu_report_exception", "gpu_signal_exception", "gpu_report_exception_name",
    "gpu_report_exception_frame")

"""
Compile kernel `f` for argument types `tt` to optimized LLVM IR for the PTX target and run
GPUCompiler's IR validation. Returns the validation errors that linking would not resolve,
and the names of the math-library functions the kernel calls.
"""
function validate_kernel(f, tt)
    target = GPUCompiler.PTXCompilerTarget(; cap=v"7.5", ptx=v"7.5")
    params = CUDA.CUDACompilerParams(v"7.5", v"7.5")
    config = GPUCompiler.CompilerConfig(target, params; kernel=true, name=nothing, always_inline=false,
        libraries=false, validate=false)
    job = GPUCompiler.CompilerJob(GPUCompiler.methodinstance(typeof(f), tt), config)
    GPUCompiler.JuliaContext() do ctx
        ir, _ = GPUCompiler.compile(:llvm, job)
        errors = GPUCompiler.check_ir!(job, GPUCompiler.IRError[], ir)
        math = String[]
        real_errors = String[]
        for (kind, bt, meta) in errors
            desc = meta === nothing ? "" : sprint(show, meta)
            m = match(r"\"([^\"]+)\"", desc)
            fname = m === nothing ? desc : m.captures[1]
            if kind == GPUCompiler.UNKNOWN_FUNCTION && startswith(fname, "__nv_")
                push!(math, replace(fname, "__nv_" => ""))
            elseif !(kind == GPUCompiler.UNKNOWN_FUNCTION && fname in EXCEPTION_RUNTIME)
                push!(real_errors, "$kind $fname at $(isempty(bt) ? "?" : first(bt))")
            end
        end
        return real_errors, sort(unique(math))
    end
end

# Tile offsets and sizes; then, after the camera arguments, bhspin, nx, ny, nmaxstep, freq, fovx, fovy.
head = (Int64, Int64, Int64, Int64)
tail = (Float64, Int64, Int64, Int64, Float64, Float64, Float64)
CamPos = SVector{4,Float64}
CamTetrad = SMatrix{4,4,Float64,16}
tt_unpol = Tuple{TrajArray,DeviceArray(Float64, 2),DeviceArray(Bool, 2),head...,CamPos,CamTetrad,tail...,Float64,Float64,Tuple{DeviceData},typeof(model)}
tt_pol = Tuple{TrajArray,DeviceArray(Float64, 2),DeviceArray(Float64, 3),DeviceArray(Bool, 2),head...,CamPos,CamTetrad,CamTetrad,tail...,Float64,Int64,Tuple{DeviceData},typeof(model)}
try
    errors_unpol, math_unpol = validate_kernel(Jipole.Imaging.raytrace_image_gpu!, tt_unpol)
    errors_pol, math_pol = validate_kernel(ext.raytrace_image_gpu_pol!, tt_pol)
    foreach(e -> println("      ", e), errors_pol)
    report(isempty(errors_unpol), "control: the existing unpolarized kernel compiles for the GPU target with no validation error")
    report(isempty(errors_pol), "the polarized kernel compiles for the GPU target with $(length(errors_pol)) validation error(s)")
    println("info  math-library functions called by the polarized kernel: ", join(math_pol, " "))
    println("info  of which the unpolarized kernel does not call: ", join(setdiff(math_pol, math_unpol), " "))
catch err
    # This part uses internals of GPUCompiler/CUDA.jl, which may change between versions.
    err isa Union{MethodError,UndefVarError,ArgumentError} || rethrow()
    println("SKIP  GPU-target compilation: ", first(sprint(showerror, err), 200))
end

println()
if isempty(failures)
    println("PASS")
else
    println("FAIL ($(length(failures)) check(s))")
    exit(1)
end
