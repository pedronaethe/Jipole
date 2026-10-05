/* Reference values for test/coeffs_pol: ipole's own thermal fit routines (src/symphony),
 * called the way jar_calc_dist calls them in src/model_radiation.c.
 *
 * Reads lines "Ne nu Thetae B theta" from standard input and prints, for each,
 *   jI jQ jV rhoQ rhoV rhoV_dexter Bnu_inv
 * where jI, jQ, jV are the Dexter (2016) emissivities (dexter_fit = 1), rhoQ and rhoV the
 * rotativities as used for a regular evaluation (dexter_fit = 0), rhoV_dexter the rotativity
 * as used for a field-aligned wavevector (dexter_fit = 1), all in the fit's own sign
 * convention and CGS units, and Bnu_inv the invariant Planck function.
 *
 * See make_reference.sh for how this is built and run.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "fits.h"
#include "params.h"
#include "radiation.h"

int main(int argc, char **argv)
{
  double Ne, nu, Thetae, B, theta;
  while (scanf("%lf %lf %lf %lf %lf", &Ne, &nu, &Thetae, &B, &theta) == 5) {
    struct parameters p;
    setConstParams(&p);
    p.electron_density = Ne; p.nu = nu; p.observer_angle = theta; p.magnetic_field = B;
    p.distribution = p.MAXWELL_JUETTNER; p.theta_e = Thetae;
    p.dexter_fit = 1;
    double jI = j_nu_fit(&p, p.STOKES_I);
    double jQ = j_nu_fit(&p, p.STOKES_Q);
    double jV = j_nu_fit(&p, p.STOKES_V);
    double rVd = rho_nu_fit(&p, p.STOKES_V);   /* Dexter branch (field-aligned early return) */
    p.dexter_fit = 0;
    double rQ = rho_nu_fit(&p, p.STOKES_Q);
    double rV = rho_nu_fit(&p, p.STOKES_V);
    double Bnuinv = Bnu_inv(nu, Thetae);
    printf("%.17g %.17g %.17g %.17g %.17g %.17g %.17g\n", jI, jQ, jV, rQ, rV, rVd, Bnuinv);
  }
  return 0;
}
