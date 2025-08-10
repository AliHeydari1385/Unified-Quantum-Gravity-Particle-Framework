import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt  # Optional for plotting

# Physical constants (SI units)
hbar = 1.0545718e-34  # J s
c = 2.99792458e8      # m/s
m_p = 1.6726219e-27   # kg (proton mass)
m_n = 1e-36           # kg (speculative neutrino mass, ~0.1 eV/c^2)
m_a = 1.78e-36        # kg (speculative axion mass, ~1e-5 eV/c^2)
G = 6.67430e-11       # m^3 kg^-1 s^-2 (gravitational constant)
omega_photon = 1e15   # rad/s (high-energy photon frequency)
k = (hbar * omega_photon)**2 / (m_p * c**2)  # Harmonic constant for photon force

# Adjusted coupling constants based on experimental data
lambda_pn = 1e-50     # Proton-neutrino, from Fermi constant G_F
lambda_pa = 1e-55     # Proton-axion, from axion-nucleon coupling g_aN
lambda_na = 1e-60     # Neutrino-axion, speculative from extended models

# Gravitational "effective mass" (speculative, as before)
M_grav = 1e-27        # kg (small for quantum scale)

# System of ODEs: state = [x_p, y_p, z_p, vx_p, vy_p, vz_p, x_n, y_n, z_n, vx_n, vy_n, vz_n, x_a, y_a, z_a, vx_a, vy_a, vz_a]
def model(state, t):
    x_p, y_p, z_p, vx_p, vy_p, vz_p, x_n, y_n, z_n, vx_n, vy_n, vz_n, x_a, y_a, z_a, vx_a, vy_a, vz_a = state
    
    r_p = np.array([x_p, y_p, z_p])
    r_n = np.array([x_n, y_n, z_n])
    r_a = np.array([x_a, y_a, z_a])
    
    # Positions relative
    r_pn = r_p - r_n
    r_pa = r_p - r_a
    r_na = r_n - r_a
    dist_pn = np.linalg.norm(r_pn) + 1e-20  # Avoid division by zero
    dist_pa = np.linalg.norm(r_pa) + 1e-20
    dist_na = np.linalg.norm(r_na) + 1e-20
    
    # Forces on proton
    F_photon_p = -k * r_p  # Harmonic photon force
    F_grav_p = -G * m_p * M_grav * r_p / (np.linalg.norm(r_p)**3 + 1e-20)
    F_couple_pn = -lambda_pn * r_pn / dist_pn**3  # From neutrino
    F_couple_pa = -lambda_pa * r_pa / dist_pa**3  # From axion
    F_p_total = F_photon_p + F_grav_p + F_couple_pn + F_couple_pa
    a_p = F_p_total / m_p
    
    # Forces on neutrino
    F_grav_n = -G * m_n * M_grav * r_n / (np.linalg.norm(r_n)**3 + 1e-20)
    F_couple_np = lambda_pn * r_pn / dist_pn**3   # Reaction from proton
    F_couple_na = -lambda_na * r_na / dist_na**3  # From axion
    F_n_total = F_grav_n + F_couple_np + F_couple_na
    a_n = F_n_total / m_n
    
    # Forces on axion
    F_grav_a = -G * m_a * M_grav * r_a / (np.linalg.norm(r_a)**3 + 1e-20)
    F_couple_ap = lambda_pa * r_pa / dist_pa**3   # Reaction from proton
    F_couple_an = lambda_na * r_na / dist_na**3   # Reaction from neutrino
    F_a_total = F_grav_a + F_couple_ap + F_couple_an
    a_a = F_a_total / m_a
    
    # Derivatives
    return [vx_p, vy_p, vz_p, a_p[0], a_p[1], a_p[2],
            vx_n, vy_n, vz_n, a_n[0], a_n[1], a_n[2],
            vx_a, vy_a, vz_a, a_a[0], a_a[1], a_a[2]]

# Initial conditions: all at origin with zero velocity
state0 = np.zeros(18)

# Time array
t = np.linspace(0, 1e-12, 1000)  # 1 picosecond, 1000 points

# Solve ODE
solution = odeint(model, state0, t, rtol=1e-10, atol=1e-10)

# Extract positions and velocities
positions_p = solution[:, 0:3]
velocities_p = solution[:, 3:6]
positions_n = solution[:, 6:9]
velocities_n = solution[:, 9:12]
positions_a = solution[:, 12:15]
velocities_a = solution[:, 15:18]

# Compute magnitudes
mag_pos_p = np.linalg.norm(positions_p, axis=1)
mag_vel_p = np.linalg.norm(velocities_p, axis=1)
mag_pos_n = np.linalg.norm(positions_n, axis=1)
mag_vel_n = np.linalg.norm(velocities_n, axis=1)
mag_pos_a = np.linalg.norm(positions_a, axis=1)
mag_vel_a = np.linalg.norm(velocities_a, axis=1)

# Results
print("Final position magnitude proton:", mag_pos_p[-1], "m")
print("Final velocity magnitude proton:", mag_vel_p[-1], "m/s")
print("Final position magnitude neutrino:", mag_pos_n[-1], "m")
print("Final velocity magnitude neutrino:", mag_vel_n[-1], "m/s")
print("Final position magnitude axion:", mag_pos_a[-1], "m")
print("Final velocity magnitude axion:", mag_vel_a[-1], "m/s")
print("Max displacement proton:", np.max(mag_pos_p), "m")
print("Max displacement neutrino:", np.max(mag_pos_n), "m")
print("Max displacement axion:", np.max(mag_pos_a), "m")

# Optional: Plot displacements
# plt.plot(t, mag_pos_p, label='Proton')
# plt.plot(t, mag_pos_n, label='Neutrino')
# plt.plot(t, mag_pos_a, label='Axion')
# plt.xlabel('Time (s)')
# plt.ylabel('Displacement (m)')
# plt.legend()
# plt.show()
