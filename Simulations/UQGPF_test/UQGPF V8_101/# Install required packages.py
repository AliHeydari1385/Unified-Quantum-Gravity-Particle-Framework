# Install required packages
!pip install camb emcee corner getdist tqdm h5py numpy scipy matplotlib pandas -q

import numpy as np
import pandas as pd
import camb
from camb import model
import emcee
from emcee.moves import StretchMove
import corner
import matplotlib.pyplot as plt
import os
import time
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interpn
from tqdm import tqdm
import scipy.optimize as opt

# Paths (same as before)
SNIA_PATH = 'D:/heydari prv/theory/GitHub/Unified-Quantum-Gravity-Particle-Framework/Simulations/UQGPF_test/Reference/Pantheon+SH0ES.dat'
COV_PATH = 'D:/heydari prv/theory/GitHub/Unified-Quantum-Gravity-Particle-Framework/Simulations/UQGPF_test/Reference/4_DISTANCES_AND_COVAR_extract/Pantheon+SH0ES_STAT+SYS.cov'
CMB_CL_PATH = 'D:/heydari prv/theory/GitHub/Unified-Quantum-Gravity-Particle-Framework/Simulations/UQGPF_test/Reference/ACT+SPT_cl.dat'
CMB_COV_PATH = 'D:/heydari prv/theory/GitHub/Unified-Quantum-Gravity-Particle-Framework/Simulations/UQGPF_test/Reference/ACT+SPT_cov.dat'
BBL_EQ_PATH = 'D:/heydari prv/theory/GitHub/Unified-Quantum-Gravity-Particle-Framework/Simulations/UQGPF_test/Reference/Bbl_148_equa_v2p2.dat'
BBL_SO_PATH = 'D:/heydari prv/theory/GitHub/Unified-Quantum-Gravity-Particle-Framework/Simulations/UQGPF_test/Reference/Bbl_148_south_v2p2.dat'
BBL_SP_PATH = 'D:/heydari prv/theory/GitHub/Unified-Quantum-Gravity-Particle-Framework/Simulations/UQGPF_test/Reference/Bbl_150_spt_v2p2.dat'

# Reduced grid size for faster precompute (8 points per dimension, total 512 points)
OMEGA_M_GRID = np.linspace(0.1, 0.5, 8)
H_GRID = np.linspace(0.5, 0.9, 8)
GAMMA_GRID = np.linspace(0.0, 3.0, 8)  # Restricted to [0,3] to avoid w>0 regions

def load_bbl_windows():
    try:
        bbl_eq = np.loadtxt(BBL_EQ_PATH)[:, 1:]
        bbl_so = np.loadtxt(BBL_SO_PATH)[:, 1:]
        bbl_sp = np.loadtxt(BBL_SP_PATH)[:, 1:]
        
        win_act = np.vstack([bbl_eq[:, 3:24].T, bbl_so[:, 3:24].T])
        win_spt = bbl_sp[:, :].T
        
        lmax_win = min(bbl_eq.shape[0], bbl_sp.shape[0])
        win_act = win_act[:, :lmax_win]
        win_spt = win_spt[:, :lmax_win]
        
        bbl_matrix = np.vstack([win_act, win_spt])
        print(f"Loaded combined Bbl matrix with shape: {bbl_matrix.shape}")
        return bbl_matrix
    except Exception as e:
        print(f"Bbl load error: {e}. Using identity.")
        return np.eye(89)

bbl_matrix = load_bbl_windows()
n_l = bbl_matrix.shape[1]

def precompute_camb_grid(lmax):
    start_time = time.time()
    total_points = len(OMEGA_M_GRID) * len(H_GRID) * len(GAMMA_GRID)
    cl_grid = np.zeros((len(OMEGA_M_GRID), len(H_GRID), len(GAMMA_GRID), lmax + 1))
    rejection_reasons = []
    success_count = 0
    
    with tqdm(total=total_points, desc="Precomputing CAMB Grid") as pbar:
        for i, om in enumerate(OMEGA_M_GRID):
            for j, h in enumerate(H_GRID):
                for k, gamma in enumerate(GAMMA_GRID):
                    try:
                        pars = camb.CAMBparams()
                        pars.set_cosmology(H0=h*100, ombh2=0.022, omch2=om*(h**2) - 0.022, tau=0.06)
                        pars.InitPower.set_params(As=2e-9, ns=0.965)
                        w = (gamma - 3) / 3
                        pars.set_dark_energy(w=w, wa=0, dark_energy_model='fluid')
                        pars.NonLinear = model.NonLinear_none
                        pars.set_accuracy(AccuracyBoost=1.0, lAccuracyBoost=1.0, lSampleBoost=1.0)
                        pars.set_for_lmax(lmax, lens_potential_accuracy=1)  # Enabled lensing
                        pars.Reion.set_tau(0.06)
                        pars.Reion.use_optical_depth = True
                        results = camb.get_results(pars)
                        cls = results.get_total_cls(lmax=lmax, CMB_unit='muK')[:, 0]
                        ells = np.arange(lmax + 1)
                        dl = (ells * (ells + 1) / (2 * np.pi)) * cls
                        cl_grid[i, j, k] = dl
                        success_count += 1
                    except camb.CAMBError as e:
                        rejection_reasons.append(f"Error for om={om:.2f}, h={h:.2f}, gamma={gamma:.2f}: {e}")
                        cl_grid[i, j, k] = np.zeros(lmax + 1)
                    pbar.update(1)
    
    np.savez('camb_grid_v8_101.npz', cl_grid=cl_grid, om_grid=OMEGA_M_GRID, h_grid=H_GRID, gamma_grid=GAMMA_GRID)
    with open('rejection_reasons_v8_101.txt', 'w') as f:
        f.write('\n'.join(rejection_reasons))
    print(f"Precompute time: {time.time() - start_time:.2f} s")
    print(f"Successful grid points: {success_count}/{total_points}")
    return cl_grid, OMEGA_M_GRID, H_GRID, GAMMA_GRID

lmax = 4000  # Kept at 4000
cl_grid, OMEGA_M_GRID, H_GRID, GAMMA_GRID = precompute_camb_grid(lmax)

def load_snia_data():
    try:
        df = pd.read_csv(SNIA_PATH, sep='\s+', header=0, on_bad_lines='skip')
        ww = (df['zHD'] > 0.001) | (df['IS_CALIBRATOR'] == 1)  # Softened filter to include more points
        snia_zcmb = df['zHD'][ww].astype(float).values
        snia_zhel = df['zHEL'][ww].astype(float).values
        snia_mb = df['m_b_corr'][ww].astype(float).values
        snia_is_cal = df['IS_CALIBRATOR'][ww].astype(int).values
        snia_ceph = df['CEPH_DIST'][ww].astype(float).values
        snia_idx = np.where(ww)[0]
        print(f"Loaded {len(snia_zcmb)} SNIa points (including {np.sum(snia_is_cal)} calibrators)")
        return snia_zcmb, snia_zhel, snia_mb, snia_is_cal, snia_ceph, snia_idx
    except Exception as e:
        print(f"SNIa load error: {e}. Using dummy.")
        dummy_z = np.linspace(0.01, 2.0, 100)
        return dummy_z, dummy_z, np.full(100, 40.0), np.zeros(100, dtype=int), np.full(100, np.nan), np.arange(100)

def load_snia_cov(snia_idx):
    try:
        with open(COV_PATH, 'r') as f:
            lines = f.readlines()
        n = int(lines[0].strip())
        cov_flat = []
        for line in lines[1:]:
            if '---' in line:
                continue
            vals = [float(x) for x in line.strip().split() if x and x != '|']
            cov_flat.extend(vals)
        cov = np.array(cov_flat).reshape((n, n))
        cov_sub = cov[np.ix_(snia_idx, snia_idx)]
        print(f"Loaded SNIa cov: {cov_sub.shape}")
        return cov_sub
    except Exception as e:
        print(f"Cov load error: {e}. Using dummy.")
        return np.diag(np.full(len(snia_idx), 0.1**2))

def load_cmb_data():
    try:
        with open(CMB_CL_PATH, 'r') as f:
            lines = f.readlines()
        data = []
        for line in lines:
            vals = [float(x) for x in line.strip().split() if x]
            if len(vals) == 3:
                data.append(vals)
        data = np.array(data)
        if len(data) != 89:
            raise ValueError(f"Expected 89 points, got {len(data)}")
        cmb_l = data[:, 0].astype(int)
        cmb_cl = data[:, 1]
        cmb_err = data[:, 2]
        
        with open(CMB_COV_PATH, 'r') as f:
            lines = f.readlines()
        cov_flat = []
        for line in lines:
            if '---' in line:
                continue
            vals = [float(x) for x in line.strip().split() if x and x != '|']
            cov_flat.extend(vals)
        if len(cov_flat) != 89 * 89:
            raise ValueError(f"Expected 7921 elements, got {len(cov_flat)}")
        cmb_cov = np.array(cov_flat).reshape((89, 89))
        
        print(f"Loaded CMB: {len(cmb_cl)} points, cov {cmb_cov.shape}")
        print("CMB cl sample:", cmb_cl[:5])
        return cmb_l, cmb_cl, cmb_cov
    except Exception as e:
        print(f"CMB load error: {e}. Using dummy.")
        dummy_l = np.arange(600, 600 + 89 * 30, 30)
        return dummy_l, np.full(89, 1e-10), np.eye(89)

cmb_l, cmb_cl_obs, cmb_cov = load_cmb_data()
snia_zcmb, snia_zhel, snia_mb_obs, snia_is_cal, snia_ceph, snia_idx = load_snia_data()
snia_cov = load_snia_cov(snia_idx)
inv_cov_snia = np.linalg.inv(snia_cov)
inv_cov_cmb = np.linalg.inv(cmb_cov)

def model_mu(zcmb, zhel, is_cal, ceph, omega_m, h, gamma, beta, alpha, m_tcalib, scale_fparam):
    mu_theory = np.full_like(zcmb, np.nan)
    mu_theory[is_cal == 1] = ceph[is_cal == 1] + m_tcalib
    mask = is_cal == 0
    z = zcmb[mask]
    zz = np.linspace(0, np.max(z) + 0.1, 300)
    Ez = np.sqrt(omega_m * (1 + zz)**3 + (1 - omega_m) * (1 + zz)**gamma)
    int_h = cumulative_trapezoid(1 / Ez, zz, initial=0)
    int_h_interp = np.interp(z, zz, int_h)
    dl = (1 + zhel[mask]) * (299792.458 / (h * 100)) * int_h_interp
    cosmo_mu = 5 * np.log10(dl) + 25
    custom_term = scale_fparam * beta * np.log10(alpha / (z + 1e-6))
    mu_theory[mask] = cosmo_mu + m_tcalib + custom_term
    return mu_theory

def get_theory_cl(omega_m, h, gamma):
    try:
        points = (OMEGA_M_GRID, H_GRID, GAMMA_GRID)
        cl_theory = interpn(points, cl_grid, (omega_m, h, gamma), method='linear', bounds_error=True, fill_value=None)
        cl_trim = cl_theory[:min(n_l, lmax + 1)]
        if len(cl_trim) < n_l:
            cl_trim = np.pad(cl_trim, (0, n_l - len(cl_trim)), mode='constant')
        cl_binned = bbl_matrix @ cl_trim
        if len(cl_binned) != len(cmb_cl_obs):
            return None
        return cl_binned
    except ValueError:
        return None
    except Exception as e:
        print(f"get_theory_cl error: {e}")
        return None

def log_likelihood(theta):
    omega_m, h, gamma, beta, alpha, m_tcalib, scale_fparam = theta
    if not all(np.isfinite(theta)) or omega_m <= 0 or h <= 0 or gamma < 0:
        return -np.inf
    try:
        mu_theory = model_mu(snia_zcmb, snia_zhel, snia_is_cal, snia_ceph, *theta)
        delta_mu = snia_mb_obs - mu_theory
        chi2_snia = np.dot(delta_mu, inv_cov_snia @ delta_mu)

        cl_theory = get_theory_cl(omega_m, h, gamma)
        if cl_theory is None:
            return -np.inf
        delta_cl = cmb_cl_obs - cl_theory
        chi2_cmb = np.dot(delta_cl, inv_cov_cmb @ delta_cl)

        chi2_total = chi2_snia + chi2_cmb
        return -0.5 * chi2_total
    except Exception as e:
        print(f"likelihood error: {e}")
        return -np.inf

def log_prior(theta):
    omega_m, h, gamma, beta, alpha, m_tcalib, scale_fparam = theta
    if (0.1 < omega_m < 0.5) and (0.5 < h < 1.0) and (0.0 < gamma < 3.0) and (0 < beta < 100) and (0 < alpha < 1000) and (-10 < m_tcalib < 10) and (-10 < scale_fparam < 10):
        return 0.0
    return -np.inf

def log_probability(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll

# Optimization to find better starting point
def neg_log_prob(theta):
    return -log_probability(theta)

ndim = 7
initial_mean = np.array([0.3, 0.7, 0.01, 45.0, 400.0, 0.0, -0.005])  # gamma initial close to 0 for w~-1
bounds = [(0.1, 0.5), (0.5, 1.0), (0.0, 3.0), (0, 100), (0, 1000), (-10, 10), (-10, 10)]
res = opt.minimize(neg_log_prob, initial_mean, method='Nelder-Mead', bounds=bounds)
if res.success:
    initial_mean = res.x
    print(f"Optimization successful. Starting from: {initial_mean}")
else:
    print(f"Optimization failed: {res.message}. Using initial guess.")

# MCMC setup
nwalkers = 128
nsteps = 3000
initial = initial_mean + 1e-4 * np.random.randn(nwalkers, ndim)

backend = emcee.backends.HDFBackend("mcmc_backend_v8_101.h5")
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, moves=[(StretchMove(a=2.0), 1.0)], backend=backend)

# Run with tqdm progress and chi2 breakdown
with tqdm(total=nsteps, desc="MCMC Progress") as pbar:
    for i, result in enumerate(sampler.sample(initial, iterations=nsteps)):
        pbar.update(1)
        if i % 100 == 0:
            mean_theta = np.mean(result.coords, axis=0)
            mean_log_prob = np.mean(result.log_prob)
            print(f"Step {i}: mean log_prob={mean_log_prob:.2e}")
            ll = log_likelihood(mean_theta)
            print(f"Chi2 from mean theta: {-2 * ll:.2e}")

# Analysis
samples = sampler.get_chain(discard=500, thin=15, flat=True)
if len(samples) < 100:
    print("Warning: Too few samples after burn-in/thinning. MCMC may not have converged.")
fig = corner.corner(samples, labels=["Omega_m", "h", "gamma", "beta", "alpha", "M_tcalib", "scale_fparam"])
fig.savefig("uqgpf_posteriors_v8_101.pdf")

median_theta = np.median(samples, axis=0)
print(f"Median parameters: {median_theta}")
mu_theory = model_mu(snia_zcmb, snia_zhel, snia_is_cal, snia_ceph, *median_theta)
residuals = snia_mb_obs - mu_theory
np.savetxt("residuals_v8_101.csv", residuals)
plt.scatter(snia_zcmb[snia_is_cal == 0], residuals[snia_is_cal == 0])
plt.xlabel('z')
plt.ylabel('Residuals')
plt.savefig("residuals_plot_v8_101.pdf")

cl_theory = get_theory_cl(*median_theta[:3])
if cl_theory is not None:
    plt.plot(cmb_l, cmb_cl_obs, label='Data')
    plt.plot(cmb_l, cl_theory, label='UQGPF Fit')
    plt.xlabel('ell')
    plt.ylabel('$D_\\ell$')
    plt.legend()
    plt.savefig("uqgpf_fit_v8_101.pdf")

# Compute final chi2 breakdown
final_ll = log_likelihood(median_theta)
print(f"Final log-likelihood: {final_ll:.2e}")
print("Done! Updated to v8.101: Restricted GAMMA_GRID and prior to [0,3] to avoid w>0 errors. Added bounds_error=True in interpn to return -inf for out-of-grid points. Added optimization (Nelder-Mead) to find better starting point before MCMC. Softened SNIa filter to z>0.001. This should help reduce chi2 and improve convergence.")
