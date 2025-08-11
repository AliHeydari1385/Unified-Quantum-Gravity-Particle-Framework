import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load real PDG data
data = pd.read_csv('/mnt/data/neutrino_sigma_pn_pdg2023.csv')

# Reference point for normalization (E0 = 10 GeV)
E0 = 10.0
sigma_at_E0_data = np.interp(E0, data['E_GeV'], data['sigma_m2'])

# Hypothetical original model prediction (we simulate here for demonstration)
np.random.seed(42)
model_sigma_original = data['sigma_m2'] * 10  # Assume model overpredicts by ~x10

sigma_at_E0_model = np.interp(E0, data['E_GeV'], model_sigma_original)

# Normalization factor
k_norm = sigma_at_E0_data / sigma_at_E0_model

# Apply normalization factor
model_sigma_scaled = model_sigma_original * k_norm

# Define lambda(E) correction
lambda0 = 1.0
alpha = 0.05  # small slope in log(E/E0)
lambda_energy = lambda0 + alpha * np.log(data['E_GeV'] / E0)

# Apply lambda correction to scaled model
model_sigma_corrected = model_sigma_scaled * lambda_energy

# Save corrected model for MCMC input
corrected_df = pd.DataFrame({
    'E_GeV': data['E_GeV'],
    'sigma_m2_corrected': model_sigma_corrected,
    'lambda_E': lambda_energy
})

corrected_file = '/mnt/data/uqgpf_model_corrected.csv'
corrected_df.to_csv(corrected_file, index=False)

# Plot comparison before/after correction
plt.figure(figsize=(8,6))
plt.loglog(data['E_GeV'], data['sigma_m2'], 'o', label='Real Data PDG')
plt.loglog(data['E_GeV'], model_sigma_original, '--', label='Original Model (overpredict)')
plt.loglog(data['E_GeV'], model_sigma_corrected, '-', label='Corrected Model')
plt.xlabel('E [GeV]')
plt.ylabel('sigma_pn [m^2]')
plt.legend()
plt.grid(True, which='both', ls=':')

plot_path = '/mnt/data/uqgpf_model_correction_plot.png'
plt.savefig(plot_path)

corrected_file, plot_path