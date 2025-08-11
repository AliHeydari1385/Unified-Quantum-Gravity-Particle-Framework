import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------
# 1. Load real PDG neutrino sigma data
# ------------------
data_file = '/mnt/data/neutrino_sigma_pn_pdg2023.csv'
data = pd.read_csv(data_file)

E = data['E_GeV'].values
sigma = data['sigma_m2'].values
sigma_err = data['uncertainty_m2'].values

# ------------------
# 2. Define UQGPF model (simplified example)
# sigma_model(E; lambda, scale) = scale * E ** lambda
# ------------------
def sigma_model(E, lam, scale):
    return scale * (E ** lam)

# ------------------
# 3. Define likelihood function (Gaussian errors)
# ------------------
def log_likelihood(params):
    lam, scale = params
    if scale <= 0:
        return -np.inf  # invalid
    model_vals = sigma_model(E, lam, scale)
    chi2 = np.sum(((sigma - model_vals) / sigma_err) ** 2)
    return -0.5 * chi2

# ------------------
# 4. MCMC: Metropolis-Hastings
# ------------------
np.random.seed(42)

n_chains = 4
n_samples = 15000
init_params = [(1.0, 2.5e-42), (0.9, 2.6e-42), (1.1, 2.4e-42), (1.05, 2.45e-42)]
step_sizes = [0.01, 1e-44]

all_samples = []
for chain in range(n_chains):
    lam, scale = init_params[chain]
    samples = []
    current_ll = log_likelihood((lam, scale))
    for i in range(n_samples):
        proposal = [np.random.normal(lam, step_sizes[0]),
                    np.random.normal(scale, step_sizes[1])]
        proposal_ll = log_likelihood(proposal)
        if np.log(np.random.rand()) < (proposal_ll - current_ll):
            lam, scale = proposal
            current_ll = proposal_ll
        samples.append([lam, scale])
    all_samples.append(samples)

all_samples = np.array(all_samples)  # shape: (chains, samples, 2)

# ------------------
# 5. Save results
# ------------------
np.save('/mnt/data/mcmc_samples_real.npy', all_samples)
pd.DataFrame(all_samples.reshape(-1, 2), columns=['lambda', 'scale']).to_csv('/mnt/data/mcmc_samples_real.csv', index=False)

# ------------------
# 6. Visualization
# ------------------
flat_samples = all_samples.reshape(-1, 2)

# Trace plots
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
for c in range(n_chains):
    axes[0].plot(all_samples[c,:,0], alpha=0.5, label=f'Chain {c+1}')
    axes[1].plot(all_samples[c,:,1], alpha=0.5, label=f'Chain {c+1}')
axes[0].set_ylabel('lambda')
axes[1].set_ylabel('scale')
axes[1].set_xlabel('Step')
axes[0].legend()
plt.tight_layout()
plt.savefig('/mnt/data/mcmc_trace_real.png')

# Posterior histograms
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].hist(flat_samples[:,0], bins=50, color='skyblue')
axes[0].set_xlabel('lambda')
axes[1].hist(flat_samples[:,1], bins=50, color='salmon')
axes[1].set_xlabel('scale')
plt.tight_layout()
plt.savefig('/mnt/data/mcmc_posteriors_real.png')

# Pair plot
posterior_df = pd.DataFrame(flat_samples, columns=['lambda', 'scale'])
sns.pairplot(posterior_df, corner=True)
plt.savefig('/mnt/data/mcmc_pairs_real.png')

# ------------------
# 7. Compute summary statistics
# ------------------
mean_lambda = np.mean(flat_samples[:,0])
std_lambda = np.std(flat_samples[:,0])
mean_scale = np.mean(flat_samples[:,1])
std_scale = np.std(flat_samples[:,1])

(mean_lambda, std_lambda, mean_scale, std_scale)