# Polarized thermal synchrotron coefficients: Jipole against ipole's own fit routines.
#
# reference.dat holds, for 650 inputs (Ne, nu, Thetae, B, theta), the values returned by the
# routines of ipole's src/symphony (written by make_reference.sh from ipole_fits.c). The inputs
# cover hot and cold plasma, including the temperatures where the Bessel functions underflow,
# weak and strong fields, and wavevectors from aligned to anti-aligned with the field.
#
# Checked here:
#   1. the fit routines of MaxwellJuettnerPol reproduce every reference value;
#   2. Polarization.thermal_jar assembles the eleven invariant coefficients from them as
#      ipole's jar_calc_dist does (signs, polarization cap, Kirchhoff's law, field-aligned case);
#   3. the ForwardDiff derivatives of all coefficients are finite everywhere, and agree with
#      central finite differences;
#   4. the Bessel-function ratios and their derivatives across the underflow.
#
# Usage: julia --project=scripts run.jl
using Jipole

const P = Jipole.Polarization
const M = Jipole.MaxwellJuettnerPol
const FD = Jipole.Iharm.ForwardDiff

# Relative agreement required with ipole's values (the two differ only in the last bits of
# the elementary functions; measured: 1.2e-13).
const RTOL = 1e-12
# Relative agreement required between AD and finite-difference derivatives.
const RTOL_AD = 1e-5
# Invariant emissivities below this are left out of the checks on the polarization cap and of
# the finite differences. The cap compares jI with sqrt(jQ^2 + jV^2); for emissivities below
# about 1e-154 the squares lose their bits or underflow, in ipole exactly as here, and the cap
# is then applied imprecisely or not at all. Emission that faint (the snapshot's is around
# 1e-45) contributes nothing to an image.
const J_MIN = 1e-100

failures = String[]
report(ok, what) = (println(ok ? "PASS  " : "FAIL  ", what); ok || push!(failures, what); ok)
relerr(a, b) = a == b ? 0.0 : abs(a - b) / max(abs(a), abs(b))

rows = [parse.(Float64, split(l)) for l in eachline(joinpath(@__DIR__, "reference.dat")) if !startswith(l, "#")]
println("$(length(rows)) reference points\n")

# ---------------------------------------------------------------------------------------------
# 1. Fit routines
# ---------------------------------------------------------------------------------------------
names = ("jI", "jQ", "jV", "rhoQ", "rhoV", "rhoV_dexter", "Bnu_inv")
worst = zeros(7)
for r in rows
    Ne, nu, Thetae, B, theta = r[1:5]
    jI, jQ, jV = M.maxwell_juettner_dexter_iqv(Ne, nu, Thetae, B, theta)
    mine = (jI, jQ, jV, M.maxwell_juettner_rho_q(Ne, nu, Thetae, B, theta), M.maxwell_juettner_rho_v(Ne, nu, Thetae, B, theta),
            M.maxwell_juettner_rho_v_dexter(Ne, nu, Thetae, B, theta), Jipole.Radiation.bnu_inv(nu, Thetae))
    for k in 1:7
        ref = r[5+k]
        # NaN and Inf must match as such (ipole returns them for jV at theta = 0, where 1/tan
        # diverges; that case never reaches the fit, see thermal_jar).
        worst[k] = max(worst[k], isfinite(ref) && isfinite(mine[k]) ? relerr(mine[k], ref) : (isequal(mine[k], ref) || (isnan(ref) && isnan(mine[k])) ? 0.0 : Inf))
    end
end
for k in 1:7
    report(worst[k] < RTOL, "$(rpad(names[k], 12)) largest relative difference from ipole $(worst[k])")
end

# ---------------------------------------------------------------------------------------------
# 2. Assembly of the invariant coefficients (ipole's jar_calc_dist, thermal case)
# ---------------------------------------------------------------------------------------------
"""The eleven coefficients, assembled from ipole's raw fit values as jar_calc_dist does."""
function expected_jar(r)
    Ne, nu, Thetae, B, theta = r[1:5]
    fI, fQ, fV, fRQ, fRV, fRVd, Bnuinv = r[6:12]
    z = 0.0
    (theta <= 0.0 || theta >= Float64(π)) && return (z, z, z, z, z, z, z, z, z, z, fRVd * nu)
    nusq = nu * nu
    jI, jQ, jU, jV = fI / nusq, -fQ / nusq, z, fV / nusq
    jP = sqrt(jQ^2 + jU^2 + jV^2)
    if jI < jP / 0.99
        f = jI / jP * 0.99
        jQ *= f; jU *= f; jV *= f
    end
    aI, aQ, aU, aV = Bnuinv > 0 ? (jI / Bnuinv, jQ / Bnuinv, jU / Bnuinv, jV / Bnuinv) : (z, z, z, z)
    return (jI, jQ, jU, jV, aI, aQ, aU, aV, fRQ * nu, z, fRV * nu)
end

let worst = 0.0, ncap = 0, naligned = 0, nzero = 0, bad_order = 0
    for r in rows
        c = P.thermal_jar(r[1:5]...)
        mine = ntuple(k -> getfield(c, k), 11)
        ref = expected_jar(r)
        worst = max(worst, maximum(relerr.(mine, ref)))
        (r[5] <= 0.0 || r[5] >= Float64(π)) && (naligned += 1)
        c.jI == 0 && (nzero += 1)
        jP = sqrt(c.jQ^2 + c.jU^2 + c.jV^2)
        c.jI > 0 && isapprox(jP, 0.99 * c.jI; rtol=1e-12) && (ncap += 1)
        # what the transfer solution relies on: aP < aI, and jP <= 0.99 jI
        c.jI > J_MIN || continue
        (jP <= 0.99 * c.jI * (1 + 1e-12) && sqrt(c.aQ^2 + c.aU^2 + c.aV^2) <= c.aI) || (bad_order += 1)
    end
    report(worst < RTOL, "thermal_jar   largest relative difference from ipole's assembly $worst")
    report(bad_order == 0, "thermal_jar   polarized emissivity and absorptivity never exceed the total ($bad_order violations)")
    println("info  of $(length(rows)) points: $naligned field-aligned, $ncap with the polarization cap active, $nzero with emissivity underflowed to zero")
    report(P.thermal_jar(0.0, 2.3e11, 10.0, 30.0, 1.0) == zero(P.PolCoeffs{Float64}), "thermal_jar   no electrons, no coefficients")
end

# ---------------------------------------------------------------------------------------------
# 3. Automatic differentiation
# ---------------------------------------------------------------------------------------------
jar_vec(x) = (c = P.thermal_jar(x[1], x[2], x[3], x[4], x[5]); [getfield(c, k) for k in 1:11])

let nonfinite = 0, worst = 0.0, nchecked = 0, worst_at = nothing
    for r in rows
        x = r[1:5]
        J = FD.jacobian(jar_vec, x)
        nonfinite += count(!isfinite, J)
        # Finite differences only where the coefficients are smooth and well above underflow:
        # away from the field-aligned branch and from the Bessel-function underflow.
        c0 = jar_vec(x)
        regular = 1e-2 < x[5] < π - 1e-2 && x[3] >= 1e-2 && c0[1] > J_MIN
        regular || continue
        for k in 1:5
            h = 1e-6 * abs(x[k])
            xp = copy(x); xm = copy(x); xp[k] += h; xm[k] -= h
            fd = (jar_vec(xp) .- jar_vec(xm)) ./ (2h)
            for i in 1:11
                scale = abs(c0[i]) / abs(x[k])          # size of a derivative of order one
                scale > 0 || continue
                # The polarization cap switching on or off between the two evaluations is a
                # kink, where a finite difference is not a derivative.
                capped(y) = isapprox(sqrt(y[2]^2 + y[4]^2), 0.99 * y[1]; rtol=1e-9)
                capped(jar_vec(xp)) == capped(jar_vec(xm)) || continue
                e = abs(J[i, k] - fd[i]) / max(abs(fd[i]), 1e-3 * scale)
                nchecked += 1
                if e > worst
                    worst = e; worst_at = (x, i, k, J[i, k], fd[i])
                end
            end
        end
    end
    report(nonfinite == 0, "AD            derivatives of the eleven coefficients are finite at all $(length(rows)) points ($nonfinite non-finite)")
    report(worst < RTOL_AD, "AD            largest relative difference from finite differences $worst ($nchecked derivatives compared)")
    worst < RTOL_AD || println("      at (x, coefficient, input, AD, FD) = $worst_at")
end

# ---------------------------------------------------------------------------------------------
# 4. Bessel-function ratios across the underflow of K_2
# ---------------------------------------------------------------------------------------------
let ok_vals = true, ok_der = true
    for z in (0.01, 1.0, 50.0, 700.0)
        r0, r1 = M.bessel_k_ratios(z)
        k0, k1, k2 = M.Bessels.besselk(0, z), M.Bessels.besselk(1, z), M.Bessels.besselk(2, z)
        ok_vals &= relerr(r0, k0 / k2) < 1e-13 && relerr(r1, k1 / k2) < 1e-13
        d = M.bessel_k_ratios(FD.Dual(z, 1.0))
        h = 1e-6 * z
        fd0 = (M.bessel_k_ratios(z + h)[1] - M.bessel_k_ratios(z - h)[1]) / (2h)
        fd1 = (M.bessel_k_ratios(z + h)[2] - M.bessel_k_ratios(z - h)[2]) / (2h)
        ok_der &= relerr(FD.partials(d[1], 1), fd0) < 1e-6 && relerr(FD.partials(d[2], 1), fd1) < 1e-6
    end
    report(ok_vals, "Bessel ratios K0/K2 and K1/K2 equal the ratios of the unscaled functions")
    report(ok_der, "Bessel ratios closed-form derivatives agree with finite differences")
    # Past the underflow of K_2 (z > 742.7, i.e. Thetae < 1.35e-3) ipole uses a ratio of 1.
    d = M.bessel_k_ratios(FD.Dual(1000.0, 1.0))
    report(M.bessel_k_ratios(1000.0) == (1.0, 1.0) && all(x -> FD.partials(x, 1) == 0.0, d),
        "Bessel ratios fall back to 1, with zero derivative, once K_2 underflows")
    d = M.bessel_k_ratios(FD.Dual(742.0, 1.0))
    report(all(x -> isfinite(FD.value(x)) && isfinite(FD.partials(x, 1)), d), "Bessel ratios and derivatives are finite at the edge of the underflow")
end

println()
if isempty(failures)
    println("PASS")
else
    println("FAIL ($(length(failures)) check(s))")
    exit(1)
end
