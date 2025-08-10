import numpy as np
import matplotlib.pyplot as plt

# Constants
G = 6.67430e-11      # m^3 kg^-1 s^-2
k_e = 8.99e9         # N m^2 C^-2
q_p = 1.602e-19      # C (proton)
q_n = 0              # Neutrino neutral
q_a = 0              # Axion neutral
m_p = 1.6726219e-27  # kg (proton rest mass)
m_n = 1e-36          # kg (neutrino, effective rest mass)
m_a = 1.78e-36       # kg (axion, hypothetical rest mass)
hbar = 1.0545718e-34 # J s
c = 2.99792458e8     # m/s

# Tuned couplings for LHC match
lambda_pn = 9.4e-37  # For sigma_pn ≈ 1e-39 m² (non-rel)
lambda_pa = 1e-45    # Below limit for sigma_pa < 1e-50 m²
lambda_na = 1e-60    # Weak

# Mediator-specific r0 (Yukawa range)
def get_r0(mediator):
    if mediator == 'weak': return 2e-18     # W/Z boson
    elif mediator == 'strong': return 1e-15  # Gluon-like (nuclear scale)
    elif mediator == 'gravity_extra': return 1e-19  # Extra dimensions
    else: return 1e-18  # Default

r0 = get_r0('weak')  # Change to 'strong' or 'gravity_extra' for different mediators

# Positions (initial, in meters)
pos_p = np.array([0.0, 0.0, 0.0])
pos_n = np.array([1e-15, 0.0, 0.0])  # Initial separation ~ fm
pos_a = np.array([0.0, 1e-15, 0.0])

# Initial velocities (relativistic: close to c for neutrino and axion)
vel_p = np.zeros(3)                  # Proton at rest (target-like)
vel_n = np.array([0.99 * c, 0.0, 0.0])  # Neutrino incoming at 0.99c
vel_a = np.array([0.0, 0.99 * c, 0.0])  # Axion incoming at 0.99c

# Time parameters
dt = 1e-26  # Even smaller dt for relativistic stability
steps = 100000
times = np.arange(0, steps * dt, dt)

# Lorentz factor
def gamma(v):
    v_mag = np.linalg.norm(v)
    return 1 / np.sqrt(1 - (v_mag / c)**2 + 1e-10)  # Avoid division by zero

# Force functions (non-relativistic forces, but applied relativistically)
def gravitational_force(m1, m2, r_vec):
    r = np.linalg.norm(r_vec)
    if r == 0: return np.zeros(3)
    return -G * m1 * m2 / r**3 * r_vec

def photonic_force(q1, q2, r_vec):
    r = np.linalg.norm(r_vec)
    if r == 0: return np.zeros(3)
    return k_e * q1 * q2 / r**3 * r_vec

def yukawa_force(lambda_c, r_vec, r0_val, attractive=True):
    r = np.linalg.norm(r_vec)
    if r == 0: return np.zeros(3)
    sign = -1 if attractive else 1
    factor = lambda_c * (1/r**2 + 1/(r0_val * r)) * np.exp(-r / r0_val)
    return sign * factor * r_vec / r

# Relativistic Schwarzschild radius (using total relativistic energy / c² as effective mass)
def schwarzschild_radius(E_total):
    m_eff = E_total / c**2
    return 2 * G * m_eff / c**2

# Simulation loop (Relativistic Velocity Verlet integration)
positions_p = [pos_p.copy()]
positions_n = [pos_n.copy()]
positions_a = [pos_a.copy()]

black_hole_formed = False

for t in range(1, steps):
    # Current positions and velocities
    pos_p_curr = positions_p[-1]
    pos_n_curr = positions_n[-1]
    pos_a_curr = positions_a[-1]
    vel_p_curr = vel_p
    vel_n_curr = vel_n
    vel_a_curr = vel_a
    
    r_pn = pos_p_curr - pos_n_curr
    r_pa = pos_p_curr - pos_a_curr
    r_na = pos_n_curr - pos_a_curr
    
    # Relativistic energies for BH check: E = γ m c²
    gamma_p = gamma(vel_p_curr)
    gamma_n = gamma(vel_n_curr)
    gamma_a = gamma(vel_a_curr)
    E_total = (gamma_p * m_p + gamma_n * m_n + gamma_a * m_a) * c**2
    Rs = schwarzschild_radius(E_total)
    if np.linalg.norm(r_pn) < Rs or np.linalg.norm(r_pa) < Rs or np.linalg.norm(r_na) < Rs:
        black_hole_formed = True
        print(f"Micro black hole formed at step {t}! Rs = {Rs:.2e} m")
        break
    
    # Calculate forces (using rest masses for forces, but relativistic application)
    F_grav_pn = gravitational_force(m_p, m_n, r_pn)
    F_grav_pa = gravitational_force(m_p, m_a, r_pa)
    F_photo_pn = photonic_force(q_p, q_n, r_pn)
    F_photo_pa = photonic_force(q_p, q_a, r_pa)
    F_couple_pn = yukawa_force(lambda_pn, r_pn, r0)
    F_couple_pa = yukawa_force(lambda_pa, r_pa, r0)
    F_p = F_grav_pn + F_grav_pa + F_photo_pn + F_photo_pa + F_couple_pn + F_couple_pa
    
    F_grav_np = -F_grav_pn
    F_grav_na = gravitational_force(m_n, m_a, r_na)
    F_photo_np = -F_photo_pn
    F_photo_na = photonic_force(q_n, q_a, r_na)
    F_couple_np = yukawa_force(lambda_pn, -r_pn, r0)
    F_couple_na = yukawa_force(lambda_na, r_na, r0)
    F_n = F_grav_np + F_grav_na + F_photo_np + F_photo_na + F_couple_np + F_couple_na
    
    F_grav_ap = -F_grav_pa
    F_grav_an = -F_grav_na
    F_photo_ap = -F_photo_pa
    F_photo_an = -F_photo_na
    F_couple_ap = yukawa_force(lambda_pa, -r_pa, r0)
    F_couple_an = yukawa_force(lambda_na, -r_na, r0)
    F_a = F_grav_ap + F_grav_an + F_photo_ap + F_photo_an + F_couple_ap + F_couple_an
    
    # Relativistic update: First half-velocity update
    # Acceleration is F perpendicular and parallel to v (relativistic)
    def relativistic_accel(F, v, m_rest):
        gam = gamma(v)
        v_mag = np.linalg.norm(v)
        if v_mag == 0: return F / m_rest
        v_unit = v / v_mag
        F_parallel = np.dot(F, v_unit) * v_unit
        F_perp = F - F_parallel
        a_parallel = (F_parallel / (gam**3 * m_rest))
        a_perp = (F_perp / (gam * m_rest))
        return a_parallel + a_perp
    
    a_p = relativistic_accel(F_p, vel_p_curr, m_p)
    a_n = relativistic_accel(F_n, vel_n_curr, m_n)
    a_a = relativistic_accel(F_a, vel_a_curr, m_a)
    
    # Update velocities (half step)
    vel_p_half = vel_p_curr + 0.5 * a_p * dt
    vel_n_half = vel_n_curr + 0.5 * a_n * dt
    vel_a_half = vel_a_curr + 0.5 * a_a * dt
    
    # Update positions
    pos_p_new = pos_p_curr + vel_p_half * dt
    pos_n_new = pos_n_curr + vel_n_half * dt
    pos_a_new = pos_a_curr + vel_a_half * dt
    
    # Recalculate forces at new positions (for full Verlet)
    r_pn_new = pos_p_new - pos_n_new
    r_pa_new = pos_p_new - pos_a_new
    r_na_new = pos_n_new - pos_a_new
    
    F_p_new = gravitational_force(m_p, m_n, r_pn_new) + gravitational_force(m_p, m_a, r_pa_new) + \
              photonic_force(q_p, q_n, r_pn_new) + photonic_force(q_p, q_a, r_pa_new) + \
              yukawa_force(lambda_pn, r_pn_new, r0) + yukawa_force(lambda_pa, r_pa_new, r0)
    
    F_n_new = -gravitational_force(m_p, m_n, r_pn_new) + gravitational_force(m_n, m_a, r_na_new) + \
              -photonic_force(q_p, q_n, r_pn_new) + photonic_force(q_n, q_a, r_na_new) + \
              yukawa_force(lambda_pn, -r_pn_new, r0) + yukawa_force(lambda_na, r_na_new, r0)
    
    F_a_new = -gravitational_force(m_p, m_a, r_pa_new) - gravitational_force(m_n, m_a, r_na_new) + \
              -photonic_force(q_p, q_a, r_pa_new) - photonic_force(q_n, q_a, r_na_new) + \
              yukawa_force(lambda_pa, -r_pa_new, r0) + yukawa_force(lambda_na, -r_na_new, r0)
    
    a_p_new = relativistic_accel(F_p_new, vel_p_half, m_p)
    a_n_new = relativistic_accel(F_n_new, vel_n_half, m_n)
    a_a_new = relativistic_accel(F_a_new, vel_a_half, m_a)
    
    # Full velocity update
    vel_p = vel_p_half + 0.5 * a_p_new * dt
    vel_n = vel_n_half + 0.5 * a_n_new * dt
    vel_a = vel_a_half + 0.5 * a_a_new * dt
    
    # Cap velocity to < c
    if np.linalg.norm(vel_p) >= c: vel_p = vel_p / np.linalg.norm(vel_p) * (c - 1e-5)
    if np.linalg.norm(vel_n) >= c: vel_n = vel_n / np.linalg.norm(vel_n) * (c - 1e-5)
    if np.linalg.norm(vel_a) >= c: vel_a = vel_a / np.linalg.norm(vel_a) * (c - 1e-5)
    
    positions_p.append(pos_p_new.copy())
    positions_n.append(pos_n_new.copy())
    positions_a.append(pos_a_new.copy())

positions_p = np.array(positions_p)
positions_n = np.array(positions_n)
positions_a = np.array(positions_a)

# Plot trajectories
plt.figure(figsize=(10, 6))
plt.plot(positions_p[:, 0], positions_p[:, 1], label='Proton')
plt.plot(positions_n[:, 0], positions_n[:, 1], label='Neutrino')
plt.plot(positions_a[:, 0], positions_a[:, 1], label='Axion')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title(f'Relativistic Trajectories with r0={r0:.2e} m (Mediator: weak)')
plt.legend()
plt.grid()
plt.show()

# Displacements
disp_p = np.linalg.norm(positions_p[-1] - positions_p[0])
disp_n = np.linalg.norm(positions_n[-1] - positions_n[0])
disp_a = np.linalg.norm(positions_a[-1] - positions_a[0])
print(f"Displacement - Proton: {disp_p:.2e} m")
print(f"Displacement - Neutrino: {disp_n:.2e} m")
print(f"Displacement - Axion: {disp_a:.2e} m")
print(f"Black hole formed? {black_hole_formed}")

# Effective cross-section (Born approx with relativistic correction: γ factor)
def yukawa_cross_section(lambda_c, r0_val, hbar_val, gamma_avg=1):
    return (4 * np.pi * lambda_c**2 * r0_val**2) / hbar_val**2 * gamma_avg  # Relativistic boost

gamma_avg = (gamma(vel_n) + gamma(vel_a)) / 2  # Average gamma for incoming particles
sigma_pn = yukawa_cross_section(lambda_pn, r0, hbar, gamma_avg)
sigma_pa = yukawa_cross_section(lambda_pa, r0, hbar, gamma_avg)
print(f"Cross-Section Proton-Neutrino: {sigma_pn:.2e} m²")
print(f"Cross-Section Proton-Axion: {sigma_pa:.2e} m²")

# Interaction rates (n=1e30 m^{-3}, v_avg ≈ c)
n_density = 1e30
v_avg = 0.99 * c
gamma_pn = n_density * sigma_pn * v_avg
gamma_pa = n_density * sigma_pa * v_avg
print(f"Rate Proton-Neutrino: {gamma_pn:.2e} s⁻¹")
print(f"Rate Proton-Axion: {gamma_pa:.2e} s⁻¹")

# Black hole formation probability (rough estimate: if min distance < Rs)
min_dist_pn = np.min(np.linalg.norm(positions_p - positions_n, axis=1))
prob_bh = 1 if min_dist_pn < Rs else 0
print(f"Estimated BH formation probability: {prob_bh} (based on min distance {min_dist_pn:.2e} m vs Rs {Rs:.2e} m)")
