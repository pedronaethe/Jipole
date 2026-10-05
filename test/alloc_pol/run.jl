# Heap allocations, type stability and GPU-readiness of the polarized transfer routines.
#
# The per-pixel work of the polarized path must not allocate: it runs once per geodesic step
# for every pixel, on plain numbers and on ForwardDiff dual numbers, and the same routines are
# called from the GPU kernel, where allocating is not possible at all.
#
# Checked here, on the tiny GRMHD dump shipped in test/data and on a thin disk:
#   1. zero bytes allocated by each new routine, for Float64 and for dual numbers;
#   2. concrete inferred return types (no dynamic dispatch);
#   3. the values passed between the routines are isbits (what a GPU kernel can hold);
#   4. the new methods introduce no method ambiguity;
#   5. the per-step source files contain no printing or error calls.
# The existing Stokes-I routines the polarized path is built on are measured too and reported,
# but they are not part of the pass/fail criterion.
#
# Usage: julia --project=scripts run.jl
using Jipole
using StaticArrays
using Test

const P = Jipole.Polarization
const M = Jipole.MaxwellJuettnerPol
const C = Jipole.Constants
const FD = Jipole.Iharm.ForwardDiff
const REPO = normpath(joinpath(@__DIR__, "..", ".."))

failures = String[]
report(ok, what) = (println(ok ? "PASS  " : "FAIL  ", what); ok || push!(failures, what); ok)

# Bytes allocated by the call f(args...), measured inside a function specialized on the
# argument types (so that nothing is boxed), after a warm-up call that takes care of compilation.
@noinline function allocated_bytes(f::F, args::Vararg{Any,N}) where {F,N}
    f(args...)
    return @allocated f(args...)
end
macro bytes(call)
    @assert call.head == :call
    return :(allocated_bytes($(map(esc, call.args)...)))
end

# ---------------------------------------------------------------------------------------------
# Setup: GRMHD problem on the tiny dump, in Float64 and in dual numbers
# ---------------------------------------------------------------------------------------------
dump = joinpath(REPO, "test", "data", "tiny_dump.h5")
MBH = 6.2e9; M_unit = C.M_UNIT_SANE; Rhigh = 20.0; Rlow = 1.0
ro = 1000.0; th = 163.0; phi = 0.0; freq = 230.0e9; fov_uas = 160.0; nx = 16; maxnstep = 8000
SourceD = 16.9e6 * C.PC
kw = (; MBH, Rhigh, Rlow, beta_crit=1.0, th_beg=1.74e-2, sigma_cut=1.0, sigma_cut_high=-1.0, M_unit, ro, th, phi, sourceD=SourceD)

model, data = redirect_stdout(devnull) do
    m = Jipole.Iharm.read_header(dump, MBH; th_beg=1.74e-2, Rlow=Rlow, Rhigh=Rhigh, beta_crit=1.0,
        sigma_cut=1.0, sigma_cut_high=-1.0, M_unit=M_unit)
    m, [Jipole.Iharm.load_data(dump, Rhigh, m)]
end
ctx = Jipole.Iharm.grmhd_context(model, data[1])
wrt = (:M_unit, :Rhigh, :Rlow, :beta_crit, :MBH, :ro, :th, :phi, :sourceD)
dp = Jipole.IharmPol.dual_problem(ctx; kw..., wrt=wrt)

"""Trace the geodesic of pixel (i, j) and return everything the integrators need."""
function setup_pixel(model, ro, th, phi, sourceD, i, j)
    T = promote_type(typeof(ro), typeof(th), typeof(phi), typeof(model.a))
    Rh = 1 + sqrt(1.0 - FD.value(model.a)^2)
    fov = sourceD / model.L_unit / C.MUAS_PER_RAD * fov_uas / ro
    Xcam = SVector{4,T}(Jipole.Camera.camera_position(ro, th, phi, model.a, model))
    traj = Vector{Jipole.GeoTypes.OfTrajGeneric{T}}(undef, maxnstep)
    frequ = freq * C.HPL / (C.ME * C.CL^2)
    nstep, _ = Jipole.Geodesics.get_pixel(traj, i, j, Xcam, fov, fov, frequ, nx, nx, model.a, Rh, model.rmax_geo, model, 0.0, 0.0)
    return (; T, traj, nstep, Xcam, fov, frequ, Rh)
end

# A pixel whose ray crosses the emitting plasma (checked below through its Faraday depth).
px = setup_pixel(model, ro, th, phi, SourceD, 7, 8)
pxd = setup_pixel(dp.model, dp.ro, dp.th, dp.phi, dp.sourceD, 7, 8)

println("\n== GRMHD model, Float64 and Dual{$(length(wrt)) partials} ==")
for (label, m, d, p) in (("Float64", model, data, px), ("Dual", dp.model, dp.data, pxd))
    T = p.T
    traj, nstep = p.traj, p.nstep
    S = P.integrate_emission_pol(traj, nstep, freq, m.a, m, d)
    report(FD.value(S[5]) > 0 && all(isfinite, FD.value.(S)), "$label: test ray has emission along it (Faraday depth $(round(FD.value(S[5]), sigdigits=4)), $nstep points)")

    # One representative step in the emitting region, for the per-step routines.
    n = findfirst(k -> Jipole.Radiation.radiating_region(traj[k-1].X, m, p.Rh) &&
                       FD.value(P.get_pol_state(traj[k-1].X, traj[k-1].Kcon, freq, m.a, m, d)[1].jI) > 0, nstep:-1:2)
    n = (nstep:-1:2)[n]
    ti, tf = traj[n], traj[n-1]
    dl_unit = m.L_unit * C.HPL / (C.ME * C.CL * C.CL)
    N0 = P.stokes_to_tensor(P.zero_tensor(T), T(1.0), T(0.2), T(-0.1), T(0.05))
    tau0 = zero(T)
    coeffs, Ucon, Bcon, Bcov = P.get_pol_state(tf.X, tf.Kcon, freq, m.a, m, d)
    s = Jipole.IharmPol.plasma_state(tf.X, tf.Kcon, m.a, m, d)
    lconn = Jipole.Geodesics.get_connection_analytic(ti.X, m.a, m)
    pol = zeros(T, P.NIMG, 1, 1)
    Image = zeros(T, 1, 1)
    sink = P.PolSink(d, pol, 0)

    # 1. allocations
    allocs = [
        ("maxwell_juettner_dexter_iqv", @bytes M.maxwell_juettner_dexter_iqv(s.Ne, s.nu, s.θe, s.B, s.θ)),
        ("maxwell_juettner_rho_q", @bytes M.maxwell_juettner_rho_q(s.Ne, s.nu, s.θe, s.B, s.θ)),
        ("maxwell_juettner_rho_v", @bytes M.maxwell_juettner_rho_v(s.Ne, s.nu, s.θe, s.B, s.θ)),
        ("thermal_jar", @bytes P.thermal_jar(s.Ne, s.nu, s.θe, s.B, s.θ)),
        ("plasma_state", @bytes Jipole.IharmPol.plasma_state(tf.X, tf.Kcon, m.a, m, d)),
        ("get_pol_state", @bytes P.get_pol_state(tf.X, tf.Kcon, freq, m.a, m, d)),
        ("push_polar", @bytes P.push_polar(N0, N0, lconn, ti.Kcon, 0.01)),
        ("evolve_stokes", @bytes P.evolve_stokes(T(1.0), T(0.2), T(-0.1), T(0.05), coeffs, tf.dl)),
        ("evolve_n", @bytes P.evolve_n(N0, tau0, tf.X, tf.Kcon, tf.dl, freq, m.a, m, d)),
        ("polarized_step", @bytes P.polarized_step(N0, tau0, ti.X, ti.Kcon, ti.Xhalf, ti.Kconhalf, tf.X, tf.Kcon, tf.dl, dl_unit, p.Rh, freq, m.a, m, d)),
        ("project_n", @bytes P.project_n(N0, traj[1].X, m.a, m)),
        ("integrate_emission_pol (whole ray)", @bytes P.integrate_emission_pol(traj, nstep, freq, m.a, m, d)),
        ("integrate_emission! with PolSink (whole pixel, both images)", @bytes Jipole.Radiation.integrate_emission!(traj, nstep, Image, 1, 1, freq, m.a, m, sink)),
    ]
    for (name, bytes) in allocs
        report(bytes == 0, "$label: $name allocates $bytes bytes")
    end

    # 2. type stability
    inferred = true
    try
        @inferred P.thermal_jar(s.Ne, s.nu, s.θe, s.B, s.θ)
        @inferred P.get_pol_state(tf.X, tf.Kcon, freq, m.a, m, d)
        @inferred P.evolve_n(N0, tau0, tf.X, tf.Kcon, tf.dl, freq, m.a, m, d)
        @inferred P.polarized_step(N0, tau0, ti.X, ti.Kcon, ti.Xhalf, ti.Kconhalf, tf.X, tf.Kcon, tf.dl, dl_unit, p.Rh, freq, m.a, m, d)
        @inferred P.project_n(N0, traj[1].X, m.a, m)
        @inferred P.integrate_emission_pol(traj, nstep, freq, m.a, m, d)
    catch err
        inferred = false
        println("      ", sprint(showerror, err))
    end
    report(inferred, "$label: return types of the transfer routines are inferred")

    # 3. isbits values
    report(isbits(coeffs) && isbits(N0) && isbits(Ucon) && isbits(s) && isbits(m) && isbits(ti),
        "$label: coefficients, coherency tensor, plasma state, model parameters and trajectory points are isbits")

    # Existing routines the polarized pixel is built on (reported, not gated).
    b1 = @bytes Jipole.Radiation.integrate_emission!(traj, nstep, Image, 1, 1, freq, m.a, m, d)
    b2 = @bytes Jipole.Geodesics.get_pixel(traj, 7, 8, p.Xcam, p.fov, p.fov, p.frequ, nx, nx, m.a, p.Rh, m.rmax_geo, m, 0.0, 0.0)
    println("info  $label: existing Radiation.integrate_emission! (Stokes I) allocates $b1 bytes, existing Geodesics.get_pixel $b2 bytes")
end

# ---------------------------------------------------------------------------------------------
# Thin disk (Float64 only: the existing thin-disk model is not differentiable)
# ---------------------------------------------------------------------------------------------
println("\n== Thin-disk model, Float64 ==")
let bhspin = 0.99, Rout = 100.0
    Rh = 1 + sqrt(1.0 - bhspin^2)
    cstartx = MVector{4,Float64}(0.0, log(Rh), 0.0, 0.0)
    cstopx = MVector{4,Float64}(0.0, log(Rout), 1.0, 2.0 * π)
    Mdot = 0.01 * 4 * π * C.GNEWT * 10.0 * C.MSUN * C.MP / 0.1 / C.CL / C.SIGMA_THOMPSON
    td = Jipole.ThinDisk.ThinDiskParams(bhspin, Rout, cstartx, cstopx, 10.0, Mdot, Rout)
    tdfreq = 2.417989e17
    Xcam = SVector{4,Float64}(Jipole.Camera.camera_position(1.0e4, 75.0, 0.0, bhspin, td))
    traj = Vector{Jipole.GeoTypes.OfTrajGeneric{Float64}}(undef, maxnstep)
    frequ = tdfreq * C.HPL / (C.ME * C.CL^2)
    nstep, _ = Jipole.Geodesics.get_pixel(traj, 20, 30, Xcam, 40.0 / 1.0e4, 40.0 / 1.0e4, frequ, 80, 80, bhspin, Rh, td.rmax_geo, td, 0.0, 0.0)
    nstop = Jipole.ThinDiskPol.disk_stop_index(traj, nstep, td)
    S = P.integrate_emission_pol(traj, nstop, tdfreq, bhspin, td, nothing)
    report(S[1] > 0 && S[2] != 0, "thin disk: test ray hits the disk (I = $(round(S[1] * tdfreq^3, sigdigits=4)) cgs)")
    pol = zeros(P.NIMG, 1, 1); Image = zeros(1, 1); sink = P.PolSink(nothing, pol, 0)
    b = @bytes Jipole.ThinDiskPol.disk_stop_index(traj, nstep, td)
    report(b == 0, "thin disk: disk_stop_index allocates $b bytes")
    b = @bytes P.integrate_emission_pol(traj, nstop, tdfreq, bhspin, td, nothing)
    report(b == 0, "thin disk: integrate_emission_pol (whole ray) allocates $b bytes")
    ok = try
        @inferred P.integrate_emission_pol(traj, nstop, tdfreq, bhspin, td, nothing); true
    catch err
        println("      ", sprint(showerror, err)); false
    end
    report(ok, "thin disk: return type of integrate_emission_pol is inferred")
    b1 = @bytes Jipole.Radiation.integrate_emission!(traj, nstep, Image, 1, 1, tdfreq, bhspin, td, nothing)
    b2 = @bytes Jipole.Radiation.integrate_emission!(traj, nstep, Image, 1, 1, tdfreq, bhspin, td, sink)
    println("info  thin disk: existing Radiation.integrate_emission! (Stokes I) allocates $b1 bytes; with PolSink (both images) $b2 bytes")
    report(b2 == b1, "thin disk: the polarized part of the pixel adds $(b2 - b1) bytes to the existing Stokes-I integration")
end

# ---------------------------------------------------------------------------------------------
# Method ambiguities and source hygiene
# ---------------------------------------------------------------------------------------------
println("\n== Methods and sources ==")
new_modules = (Jipole.MaxwellJuettnerPol, Jipole.Polarization, Jipole.ImagingPol, Jipole.IharmPol, Jipole.ThinDiskPol)
amb = Test.detect_ambiguities(Jipole; recursive=true)
ours = [a for a in amb if any(m -> m.module in new_modules, a)]
foreach(a -> println("      ", a[1], "\n      ", a[2]), ours)
report(isempty(ours), "no method ambiguity involves the new modules ($(length(amb)) in Jipole as a whole)")

# Code that runs once per geodesic step must not print, warn or raise.
for file in ("src/maxwell_juettner_pol.jl", "src/polarization.jl")
    code = join(filter(l -> !startswith(strip(l), "#"), split(replace(read(joinpath(REPO, file), String), r"\"\"\".*?\"\"\""s => ""), '\n')), '\n')
    bad = [m.match for m in eachmatch(r"\b(println|print|@warn|@error|@info|error|throw)\b\s*\(?", code)]
    report(isempty(bad), "$file has no print/warn/error calls" * (isempty(bad) ? "" : " (found: $(join(unique(bad), ", ")))"))
end

println()
if isempty(failures)
    println("PASS")
else
    println("FAIL ($(length(failures)) check(s))")
    exit(1)
end
