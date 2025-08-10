import numpy as np
import matplotlib.pyplot as plt

# Constants
G = 6.67430e-11  # m^3 kg^-1 s^-2
k_e = 8.99e9     # N m^2 C^-2 (for photonic, approximated as Coulomb-like)
q_p = 1.602e-19  # C (proton charge)
q_n = 0          # Neutrino neutral
q_a = 0          # Axion neutral (hypothetical)

m_p = 1.6726219e-27  # kg (proton)
m_n = 1e-36          # kg (neutrino, effective)
m_a = 1.78e-36       # kg (axion, hypothetical)

lambda_pn = 1e-50    # Coupling constants
lambda_pa = 1e-55
lambda_na = 1e-60

r0 = 1e-18           # Yukawa range parameter (m)

# Positions (initial, in meters, small scale for particle physics)
pos_p = np.array([0.0, 0.0, 0.0])
pos_n = np.array([1e-15, 0.0, 0.0])  # Initial separation ~ femtometer
pos_a = np.array([0.0, 1e-15, 0.0])

# Velocities (initially zero)
vel_p = np.zeros(3)
vel_n = np.zeros(3)
vel_a = np.zeros(3)

# Time parameters
dt = 1e-24  # Small time step for high-energy scales (s)
steps = 100000
times = np.arange(0, steps * dt, dt)

# Force functions
def gravitational_force(m1, m2, r_vec):
    r = np.linalg.norm(r_vec)
    if r == 0: return np.zeros(3)
    return -G * m1 * m2 / r**3 * r_vec  # Attractive

def photonic_force(q1, q2, r_vec):  # Approximated as Coulomb
    r = np.linalg.norm(r_vec)
    if r == 0: return np.zeros(3)
    return k_e * q1 * q2 / r**3 * r_vec  # Repulsive for same sign

def yukawa_force(lambda_c, r_vec, attractive=True):
    r = np.linalg.norm(r_vec)
    if r == 0: return np.zeros(3)
    sign = -1 if attractive else 1
    factor = lambda_c * (1/r**2 + 1/(r0 * r)) * np.exp(-r / r0)
    return sign * factor * r_vec / r  # Direction along r_vec

# Simulation loop (Verlet integration)
positions_p = [pos_p.copy()]
positions_n = [pos_n.copy()]
positions_a = [pos_a.copy()]

acc_p_prev = np.zeros(3)
acc_n_prev = np.zeros(3)
acc_a_prev = np.zeros(3)

for t in range(1, steps):
    # Compute separations
    r_pn = pos_p - pos_n
    r_pa = pos_p - pos_a
    r_na = pos_n - pos_a
    
    # Forces on proton
    F_grav_pn = gravitational_force(m_p, m_n, r_pn)
    F_grav_pa = gravitational_force(m_p, m_a, r_pa)
    F_photo_pn = photonic_force(q_p, q_n, r_pn)
    F_photo_pa = photonic_force(q_p, q_a, r_pa)
    F_couple_pn = yukawa_force(lambda_pn, r_pn)
    F_couple_pa = yukawa_force(lambda_pa, r_pa)
    F_p = F_grav_pn + F_grav_pa + F_photo_pn + F_photo_pa + F_couple_pn + F_couple_pa
    
    # Forces on neutrino
    F_grav_np = -F_grav_pn
    F_grav_na = gravitational_force(m_n, m_a, r_na)
    F_photo_np = -F_photo_pn
    F_photo_na = photonic_force(q_n, q_a, r_na)
    F_couple_np = yukawa_force(lambda_pn, -r_pn)  # Opposite
    F_couple_na = yukawa_force(lambda_na, r_na)
    F_n = F_grav_np + F_grav_na + F_photo_np + F_photo_na + F_couple_np + F_couple_na
    
    # Forces on axion
    F_grav_ap = -F_grav_pa
    F_grav_an = -F_grav_na
    F_photo_ap = -F_photo_pa
    F_photo_an = -F_photo_na
    F_couple_ap = yukawa_force(lambda_pa, -r_pa)
    F_couple_an = yukawa_force(lambda_na, -r_na)
    F_a = F_grav_ap + F_grav_an + F_photo_ap + F_photo_an + F_couple_ap + F_couple_an
    
    # Accelerations
    acc_p = F_p / m_p
    acc_n = F_n / m_n
    acc_a = F_a / m_a
    
    # Verlet update
    pos_p += vel_p * dt + 0.5 * acc_p_prev * dt**2
    pos_n += vel_n * dt + 0.5 * acc_n_prev * dt**2
    pos_a += vel_a * dt + 0.5 * acc_a_prev * dt**2
    
    vel_p += 0.5 * (acc_p_prev + acc_p) * dt
    vel_n += 0.5 * (acc_n_prev + acc_n) * dt
    vel_a += 0.5 * (acc_a_prev + acc_a) * dt
    
    acc_p_prev, acc_n_prev, acc_a_prev = acc_p, acc_n, acc_a
    
    positions_p.append(pos_p.copy())
    positions_n.append(pos_n.copy())
    positions_a.append(pos_a.copy())

# Convert to arrays
positions_p = np.array(positions_p)
positions_n = np.array(positions_n)
positions_a = np.array(positions_a)

# Plot trajectories (x-y plane for simplicity)
plt.figure(figsize=(10, 6))
plt.plot(positions_p[:, 0], positions_p[:, 1], label='Proton')
plt.plot(positions_n[:, 0], positions_n[:, 1], label='Neutrino')
plt.plot(positions_a[:, 0], positions_a[:, 1], label='Axion')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('Particle Trajectories with Yukawa Coupling')
plt.legend()
plt.grid()
plt.show()

# Calculate displacements
disp_p = np.linalg.norm(positions_p[-1] - positions_p[0])
disp_n = np.linalg.norm(positions_n[-1] - positions_n[0])
disp_a = np.linalg.norm(positions_a[-1] - positions_a[0])
print(f"Displacement - Proton: {disp_p:.2e} m")
print(f"Displacement - Neutrino: {disp_n:.2e} m")
print(f"Displacement - Axion: {disp_a:.2e} m")

# Effective cross-section for Yukawa (Fourier transform approx)
def yukawa_cross_section(lambda_c, mu, k):  # k = momentum transfer
    return (4 * np.pi * lambda_c**2 / (1 + (k * r0)**2)**2) / mu**2  # Simplified

# Example: for proton-neutrino, mu = reduced mass, k ~ 1/r0
mu_pn = (m_p * m_n) / (m_p + m_n)
k = 1 / r0  # Approx for high-energy
sigma_pn_yukawa = yukawa_cross_section(lambda_pn, mu_pn, k)
print(f"Effective Cross-Section Proton-Neutrino (Yukawa): {sigma_pn_yukawa:.2e} m²")
