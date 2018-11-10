import numpy as np

# General cosmological constants for Milky Way
G = 43007.1 # Gravitational constant in [(km/s)**2*(kpc/(1e10*M_s))]
H0 = 0.07 # Hubble in [(km/s)/kpc]
r0 = 8 # Position of the sun in [kpc]
r_vir = 213.5 # r200 for MW in [kpc] (taken from 1606.04898)

# Parameters for MW NFW profile
rho_c = 3*H0**2/(8*np.pi*G) # Critical density in [1e10*M_s/kpc**3]
delta_c_pf = 200. # Prefactor for the characteristic overdensity delta_c

# Parameters for MW Einasto profile
# Use conversion [1M_s/pc^3] = 37.96[GeV/cm**3], 
# i.e. [1e10*M_s/kpc**3] = 379.6[GeV/cm**3]
rho_s = 0.4/379.6 # Einasto_MW local density at r0 in [1e10M_s/kpc**3]

