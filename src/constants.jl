
"""
Physical and numerical constants used throughout Jipole.

All quantities are expressed in CGS units unless otherwise noted.
"""
module Constants

"""
Number of spacetime dimensions.
"""
const NDIM = 4

"""
Small number used to prevent division by zero and other numericalsingularities.
"""
const SMALL = 1e-40

"""
Planck constant (erg s).
"""
const HPL = 6.6260693e-27

"""
Electron mass (g).
"""
const ME = 9.1093826e-28

"""
Newton's gravitational constant (cm³ g⁻¹ s⁻²).
"""
const GNEWT = 6.6742e-8

"""
Speed of light in vacuum (cm s⁻¹).
"""
const CL = 2.99792458e10

"""
Solar mass (g).
"""
const MSUN = 1.989e33

"""
Microarcseconds per radian.
"""
const MUAS_PER_RAD = 2.06265e11

"""
Parsec (cm).
"""
const PC = 3.085678e18

"""
Stefan–Boltzmann constant (erg cm⁻² s⁻¹ K⁻⁴).
"""
const SIG = 5.670400e-5

"""
Boltzmann constant (erg K⁻¹).
"""
const KBOL = 1.3806505e-16

"""
Jansky (erg cm⁻² s⁻¹ Hz⁻¹).
"""
const JY = 1.0e-23

"""
Thomson scattering cross section (cm²).
"""
const SIGMA_THOMPSON = 6.65245873e-25

"""
Proton mass (g).
"""
const MP = 1.67262171e-24

"""
Elementary charge (statC).
"""
const EE = 4.80320680e-10

end


