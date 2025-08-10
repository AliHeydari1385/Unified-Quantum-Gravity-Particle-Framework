import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt  # Optional for plotting,- معادلات حرکت به صورت vectorial برای هر ذره تعریف شدند.
 if desired

# Physical constants (precise values, no simplification)
hbar = 1.0545718e-34  # J s
c = 3e8  # m/s
m_p = 1.6726219e-27  # kg (proton mass)
m_n = 1e-36  # kg (approximate neutrino mass, very small)
m_a = 1.78e-36  # kg (approximate axion mass, converted from ~10^{-5} eV/c²)
G = 6.67430e-11  # m³ kg⁻¹ s⁻² (gravitational constant)
omega_photon = 1e15  # Hz (photon frequency)
k = (hbar * omega_photon)**2 / (m_p * c**2)  # Harmonic constant for photon potential
lambda_couple = 1e-10  # Speculative coupling constant for interactions (weak)
r_scale = 1e-15  # m (scale for distances, femtometer scale)
M_grav = m_p  # Effective gravitational mass (small for quantum scale, speculative)

# Function to compute the total forces and derivatives
def equations(state, t):
    # Unpack state: [r_p_x, r_p_y, r_p_z, v_p_x, v_p_y, v_p_z, r_n_x, r_n_y, r_n_z, v_n_x, v_n_y, v_n_z, r_a_x, r_a_y, r_a_z, v_a_x, v_a_y, v_a_z]
    r_p = state[0:3]
    v_p = state[3:6]
    r_n = state[6:9]
    v_n = state[9:12]
    r_a = state[12:15]
    v_a = state[15:18]
    
    # Compute distances (vectors and magnitudes, with epsilon to avoid division by zero)
    epsilon = 1e-20
    r_pn = r_p - r_n
    dist_pn = np.linalg.norm(r_pn) + epsilon
    r_pa = r_p - r_a
    dist_pa = np.linalg.norm(r_pa) + epsilon
    r_na = r_n - r_a
    dist_na = np.linalg.norm(r_na) + epsilon
    
    # Forces on proton
    F_photon_p = -k * r_p  # Harmonic photon force on proton (vector)
    F_grav_p = -G * m_p * M_grav * r_p / (np.linalg.norm(r_p)**3 + epsilon)  # Gravity on proton
    F_couple_pn = -lambda_couple * r_pn / dist_pn**3  # Coupled force proton-neutrino
    F_couple_pa = -lambda_couple * r_pa / dist_pa**3  # Coupled force proton-axion
    F_total_p = F_photon_p + F_grav_p + F_couple_pn + F_couple_pa
    
    # Forces on neutrino
    F_grav_n = -G * m_n * M_grav * r_n / (np.linalg.norm(r_n)**3 + epsilon)  # Gravity on neutrino
    F_couple_np = -F_couple_pn  # Action-reaction for proton-neutrino
    F_couple_na = -lambda_couple * r_na / dist_na**3  # Coupled force neutrino-axion
    F_total_n = F_grav_n + F_couple_np + F_couple_na
    
    # Forces on axion
    F_grav_a = -G * m_a * M_grav * r_a / (np.linalg.norm(r_a)**3 + epsilon)  # Gravity on axion
    F_couple_ap = -F_couple_pa  # Action-reaction for proton-axion
    F_couple_an = -F_couple_na  # Action-reaction for neutrino-axion
    F_total_a = F_grav_a + F_couple_ap + F_couple_an
    
    # Derivatives
    dr_p_dt = v_p
    dv_p_dt = F_total_p / m_p
    dr_n_dt = v_n
    dv_n_dt = F_total_n / m_n
    dr_a_dt = v_a
    dv_a_dt = F_total_a / m_a
    
    # Pack derivatives
    return np.concatenate((dr_p_dt, dv_p_dt, dr_n_dt, dv_n_dt, dr_a_dt, dv_a_dt))

# Initial conditions
r_p0 = np.array([1e-15, 0, 0])  # m
v_p0 = np.array([0, 0, 0])  # m/s
r_n0 = np.array([0, 5e-16, 0])  # m
v_n0 = np.array([0, 0, 0])  # m/s
r_a0 = np.array([0, 0, 3.33e-16])  # m
v_a0 = np.array([0, 0, 0])  # m/s
state0 = np.concatenate((r_p0, v_p0, r_n0, v_n0, r_a0, v_a0))

# Time array
t = np.linspace(0, 1e-12, 10000)  # From 0 to 1e-12 s, 10000 points

# Solve ODEs with high precision
solution = odeint(equations, state0, t, rtol=1e-10, atol=1e-10)

# Extract final positions and velocities
final_state = solution[-1]
final_r_p = final_state[0:3]
final_v_p = final_state[3:6]
final_r_n = final_state[6:9]
final_v_n = final_state[9:12]
final_r_a = final_state[12:15]
final_v_a = final_state[15:18]

# Compute magnitudes
mag_final_r_p = np.linalg.norm(final_r_p)
mag_final_v_p = np.linalg.norm(final_v_p)
mag_final_r_n = np.linalg.norm(final_r_n)
mag_final_v_n = np.linalg.norm(final_v_n)
mag_final_r_a = np.linalg.norm(final_r_a)
mag_final_v_a = np.linalg.norm(final_v_a)

# Max displacements (over time)
max_disp_p = np.max(np.linalg.norm(solution[:, 0:3], axis=1))
max_disp_n = np.max(np.linalg.norm(solution[:, 6:9], axis=1))
max_disp_a = np.max(np.linalg.norm(solution[:, 12:15], axis=1))

# Print results (matching the described output)
print("بزرگی موقعیت نهایی پروتون:", mag_final_r_p, "m")
print("بزرگی سرعت نهایی پروتون:", mag_final_v_p, "m/s")
print("بزرگی موقعیت نهایی نوترینو:", mag_final_r_n, "m")
print("بزرگی سرعت نهایی نوترینو:", mag_final_v_n, "m/s")
print("بزرگی موقعیت نهایی اکسیون:", mag_final_r_a, "m")
print("بزرگی سرعت نهایی اکسیون:", mag_final_v_a, "m/s")
print("حداکثر جابجایی پروتون:", max_disp_p, "m")
print("حداکثر جابجایی نوترینو:", max_disp_n, "m")
print("حداکثر جابجایی اکسیون:", max_disp_a, "m")

# Optional: Plot trajectories if desired
# plt.plot(t, np.linalg.norm(solution[:, 0:3], axis=1), label='Proton displacement')
# plt.plot(t, np.linalg.norm(solution[:, 6:9], axis=1), label='Neutrino displacement')
# plt.plot(t, np.linalg.norm(solution[:, 12:15], axis=1), label='Axion displacement')
# plt.xlabel('Time (s)')
# plt.ylabel('Displacement (m)')
# plt.legend()
# plt.show()
