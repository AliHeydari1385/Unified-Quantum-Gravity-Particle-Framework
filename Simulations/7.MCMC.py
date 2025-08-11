import numpy as np
import matplotlib.pyplot as plt

# ------------------- شبیه‌ساز پارامتری از 6.lamda correction 4.py -------------------
# ثابت‌ها
G = 6.67430e-11
c = 2.99792458e8
hbar = 1.0545718e-34
n_density = 1e30
v_avg = 0.99 * c

# توابع مورد نیاز
def gamma_rel(v):
    return 1.0 / np.sqrt(1 - (v/c)**2 + 1e-12)

def yukawa_cross_section(lambda_c, r0_val, hbar_val, gamma_avg=1.0):
    return (4 * np.pi * lambda_c**2 * r0_val**2) / hbar_val**2 * gamma_avg

def simulate_pn(lambda_pn, lambda_pa, r0, m_n):
    # فرض سرعت نسبیتی
    gamma_avg = (gamma_rel(0.99*c) + gamma_rel(0.99*c)) / 2

    # محاسبه مقاطع مؤثر
    sigma_pn = yukawa_cross_section(lambda_pn, r0, hbar, gamma_avg)
    sigma_pa = yukawa_cross_section(lambda_pa, r0, hbar, gamma_avg)

    # نرخ برخوردها
    gamma_pn = n_density * sigma_pn * v_avg

    # فاصله مینیمم (فرضی)
    Rs = 2 * G * (1.6726219e-27 + m_n) / c**2
    r_min = 1e-15  # برای داده مصنوعی ثابت می‌گیریم

    return sigma_pn, gamma_pn, r_min

# ------------------- ساخت داده مصنوعی -------------------
true_params = [9.4e-37, 1e-45, 2e-18, 1e-36]  # λpn, λpa, r0, mn
true_outputs = simulate_pn(*true_params)
noise_levels = [true_outputs[0]*0.02, true_outputs[1]*0.02, 1e-17]  # نویز کم
observed = [np.random.normal(mu, sigma) for mu, sigma in zip(true_outputs, noise_levels)]

# ------------------- تابع لگاریتم-احتمال -------------------
def log_likelihood(params):
    lp, la, r0, mn = params
    if lp <= 0 or la <= 0 or r0 <= 0 or mn <= 0:
        return -np.inf
    model = simulate_pn(lp, la, r0, mn)
    chi2 = sum(((o - m)/s)**2 for o, m, s in zip(observed, model, noise_levels))
    return -0.5 * chi2

# ------------------- MCMC متروپولیس–هیستینگز -------------------
np.random.seed(42)
chains = 4
steps = 15000
init_guesses = [
    [1e-36, 5e-46, 1.5e-18, 5e-37],
    [8e-37, 2e-45, 2.5e-18, 2e-36],
    [5e-37, 8e-46, 1.8e-18, 8e-37],
    [1.1e-36, 9e-46, 1.9e-18, 1.2e-36]
]

all_samples = []
for c_idx in range(chains):
    current_params = np.array(init_guesses[c_idx])
    current_loglike = log_likelihood(current_params)
    samples = []
    for step in range(steps):
        proposal = current_params * np.exp(np.random.normal(0, 0.02, size=4))
        proposal_loglike = log_likelihood(proposal)
        if np.log(np.random.rand()) < proposal_loglike - current_loglike:
            current_params = proposal
            current_loglike = proposal_loglike
        samples.append(current_params)
    all_samples.append(np.array(samples))

# ------------------- پردازش نتایج -------------------
all_samples_arr = np.concatenate(all_samples, axis=0)
means = np.mean(all_samples_arr, axis=0)
stds = np.std(all_samples_arr, axis=0)
param_names = [r"lambda_pn", r"lambda_pa", r"r0", r"m_n"]

for name, mu, sd, true_val in zip(param_names, means, stds, true_params):
    print(f"{name}: {mu:.4e} ± {sd:.2e} (true = {true_val:.3e})")

# ------------------- Trace Plot -------------------
fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
for p in range(4):
    for c_idx in range(chains):
        axes[p].plot(all_samples[c_idx][:, p], alpha=0.5, label=f'Chain {c_idx+1}' if p == 0 else "")
    axes[p].set_ylabel(param_names[p])
axes[-1].set_xlabel('Step')
axes[0].legend()
plt.tight_layout()
plt.savefig('/mnt/data/mcmc_trace.png')

# ------------------- Pair Plot ساده -------------------
fig, axes = plt.subplots(4, 4, figsize=(10, 10))
for i in range(4):
    for j in range(4):
        if i == j:
            axes[i, j].hist(all_samples_arr[:, i], bins=30, color='skyblue')
            axes[i, j].axvline(true_params[i], color='r', linestyle='--')
        else:
            axes[i, j].scatter(all_samples_arr[::50, j], all_samples_arr[::50, i], s=5, alpha=0.5)
            axes[i, j].axvline(true_params[j], color='r', linestyle='--', lw=0.5)
            axes[i, j].axhline(true_params[i], color='r', linestyle='--', lw=0.5)
        if i == 3:
            axes[i, j].set_xlabel(param_names[j])
        if j == 0:
            axes[i, j].set_ylabel(param_names[i])
plt.tight_layout()
plt.savefig('/mnt/data/mcmc_pairs.png')

'/mnt/data/mcmc_trace.png', '/mnt/data/mcmc_pairs.png'