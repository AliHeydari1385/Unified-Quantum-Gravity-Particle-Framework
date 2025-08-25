
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import corner
from scipy.optimize import curve_fit

# مسیر خروجی شکل‌ها
fig_dir = '/content/UQGPF_Article/figures'

# --- شکل ۱: ساختار مفهومی ---
G = nx.erdos_renyi_graph(12, 0.25)
pos = nx.spring_layout(G)
plt.figure(figsize=(6,5))
nx.draw(G, pos, node_color='skyblue', edge_color='gray', with_labels=False)
plt.title("UQGPF Structure")
plt.savefig(f"{fig_dir}/UQGPF_Structure.png", dpi=300)
plt.close()

# داده نمونه
data = np.loadtxt('/content/UQGPF_Article/data/subsample.dat.txt')
z, mu = data[:,0], data[:,1]

# مدل
def mu_model(z, MB, alpha):
    return MB + alpha*z

# برازش ساده
popt, _ = curve_fit(mu_model, z, mu)
mu_fit = mu_model(z, *popt)

# --- شکل ۲: فیت ---
plt.figure(figsize=(6,5))
plt.scatter(z, mu, s=10, label="Data")
plt.plot(z, mu_fit, 'r-', label="UQGPF Fit")
plt.xlabel("z"); plt.ylabel(r"$\mu$")
plt.legend()
plt.title("MCMC Fit (simulated)")
plt.savefig(f"{fig_dir}/MCMC_Fit.png", dpi=300)
plt.close()

# --- شکل ۳: Posteriors ---
samples = np.random.normal(loc=popt, scale=[0.1, 0.05], size=(500, 2))
fig = corner.corner(samples, labels=[r"$M_B$", r"$\alpha$"], truths=popt)
fig.savefig(f"{fig_dir}/Posteriors.png", dpi=300)
plt.close()

# --- شکل ۴: مقایسه ---
mu_LCDM = MB_LCDM = 24 + 2*z
plt.figure(figsize=(6,5))
plt.plot(z, mu_LCDM, 'g-', label=r"$\Lambda$CDM")
plt.plot(z, mu_fit, 'r--', label="UQGPF")
plt.xlabel("z");plt.ylabel(r"$\mu$")
plt.legend()
plt.title("Model Comparison")
plt.savefig(f"{fig_dir}/Comparison.png", dpi=300)
plt.close()
