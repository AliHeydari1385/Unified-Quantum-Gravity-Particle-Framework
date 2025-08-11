import pandas as pd
import matplotlib.pyplot as plt

# Load the full-range results CSV
df = pd.read_csv('/mnt/data/uqgpf_model_corrected_fullrange.csv')

# Define regimes
def classify_regime(E):
    if E < 1.5:
        return 'QE (<1.5 GeV)'
    elif E < 5:
        return 'RES (1.5–5 GeV)'
    else:
        return 'DIS (≥5 GeV)'

df['regime'] = df['E_GeV'].apply(classify_regime)

# Assume λ ~ predicted/observed ratio for demonstration
# If real λ values are available, they should be used instead
df['lambda'] = df['uqgpf_sigma'] / df['sigma_m2']

# Plot λ vs E
plt.figure(figsize=(10,5))
for regime, group in df.groupby('regime'):
    plt.scatter(group['E_GeV'], group['lambda'], label=regime)
plt.axhline(1.0, color='k', linestyle='--', linewidth=0.8)
plt.xlabel('Neutrino Energy $E_\nu$ [GeV]')
plt.ylabel('$\\lambda$')
plt.title('Energy dependence of $\\lambda$ (UQGPF corrected model)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('/mnt/data/uqgpf_lambda_vs_E_regimes.png', dpi=300)
plt.close()

# Plot σ vs E
plt.figure(figsize=(10,5))
for regime, group in df.groupby('regime'):
    plt.scatter(group['E_GeV'], group['uqgpf_sigma'], label=f'Model – {regime}', marker='o')
    plt.errorbar(group['E_GeV'], group['sigma_m2'], yerr=group['uncertainty_m2'], fmt='x', label=f'Data – {regime}')
plt.xlabel('Neutrino Energy $E_\nu$ [GeV]')
plt.ylabel('Cross Section $\\sigma_{pn}$ [m$^2$]')
plt.title('Model vs Data $\\sigma_{pn}$ with regimes')
plt.yscale('log')
plt.legend()
plt.grid(True, which='both', alpha=0.3)
plt.savefig('/mnt/data/uqgpf_sigma_vs_E_regimes.png', dpi=300)
plt.close()

'/mnt/data/uqgpf_lambda_vs_E_regimes.png', '/mnt/data/uqgpf_sigma_vs_E_regimes.png'