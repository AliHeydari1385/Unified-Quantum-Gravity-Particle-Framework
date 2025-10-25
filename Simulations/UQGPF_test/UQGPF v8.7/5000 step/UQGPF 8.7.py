import os
import zipfile
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import emcee
import corner
from scipy.integrate import quad
from multiprocessing import Pool  # optional; کامنت کنید اگر در Colab مشکل داشت

# گام ۱: جستجوی پویا و استخراج فایل‌های ZIP (همان قبلی)
def extract_zip(zip_path, extract_to):
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted {zip_path} to {extract_to}")

zip_files = glob.glob('**/actlite_3yr_v2p2.zip', recursive=True) + glob.glob('**/Reference.zip', recursive=True)
for zip_path in zip_files:
    extract_dir = zip_path.replace('.zip', '_extract')
    extract_zip(zip_path, extract_dir)

cl_path = glob.glob('**/ACT+SPT_cl.dat', recursive=True)[0]
cov_path = glob.glob('**/ACT+SPT_cov.dat', recursive=True)[0]
snia_path = glob.glob('**/ES_AND_COVARPantheon%2BSH0ES.dat.txt', recursive=True)[0]
print(f"Found: cl={cl_path}, cov={cov_path}, snia={snia_path}")

# گام ۲: توابع بارگذاری داده‌ها (همان قبلی با sep='\s+')
def load_cov_custom(file_path):
    flat_data = []
    expected_size = 89
    expected_elements = expected_size * expected_size

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip().replace('|', '').replace('---', '')
            if not line or line.startswith('#'):
                continue
            try:
                row = [float(x) for x in line.split() if x]
                flat_data.extend(row)
            except ValueError:
                print(f"Skipping invalid line: {line}")
                continue

    num_elements = len(flat_data)
    print(f"Loaded {num_elements} elements from cov file (expected ~{expected_elements})")

    if num_elements != expected_elements:
        if num_elements > expected_elements:
            flat_data = flat_data[:expected_elements]
            print("Warning: Truncated extra elements")
        else:
            flat_data.extend([0.0] * (expected_elements - num_elements))
            print("Warning: Padded with zeros to reach expected size")

    cov = np.array(flat_data).reshape(expected_size, expected_size)
    cov = (cov + cov.T) / 2
    cov += np.eye(cov.shape[0]) * 1e-10
    inv_cov = np.linalg.inv(cov)
    print(f"Loaded cov matrix: shape={cov.shape}")
    return cov, inv_cov

def load_act_spt_data(cl_path, cov_path):
    cl_data = np.loadtxt(cl_path)
    ell = cl_data[:, 0]
    cl_obs = cl_data[:, 1]
    cl_err = cl_data[:, 2] if cl_data.shape[1] > 2 else np.sqrt(np.diag(load_cov_custom(cov_path)[0]))
    cov, inv_cov = load_cov_custom(cov_path)
    return ell, cl_obs, cov, inv_cov

def load_snia_data(file_path):
    data = pd.read_csv(file_path, sep='\s+', on_bad_lines='skip', engine='python')
    z = data.get('zHD', data.get('z', None)).values
    mu_obs = data.get('MU_SH0ES', data.get('mu', None)).values
    mu_err = data.get('MU_SH0ES_ERR_DIAG', data.get('err', None)).values
    print(f"Loaded SNIa: {len(z)} points")
    return z, mu_obs, mu_err

ell, cl_obs, cov_cmb, inv_cov_cmb = load_act_spt_data(cl_path, cov_path)
z_snia, mu_obs_snia, mu_err_snia = load_snia_data(snia_path)

# گام ۳: مدل UQGPF (بهبودیافته)
def model_uqgpf(ell, theta):
    Omega_m, h, gamma, beta, alpha = theta
    # فرم بهبودیافته Cl (placeholder مبتنی بر lensed CMB با اصلاح UQGPF)
    cl_base = (ell * (ell + 1) / (2 * np.pi)) * (1 + gamma * np.log(1 + ell / 1000))  # base power spectrum
    cl_model = cl_base * (1 + beta * np.exp(-ell / alpha))  # اصلاح با β و α
    return cl_model

def mu_theory(z, theta):
    Omega_m, h, gamma, beta, alpha = theta
    def H(x):
        # فرم UQGPF برای H(z): matter + modified dark energy
        return h * np.sqrt(Omega_m * (1 + x)**3 + (1 - Omega_m) * (1 + x)**(3 * (1 + gamma)) * np.exp(beta / alpha))

    def integrand(x):
        return 1 / H(x)

    dl = np.array([quad(integrand, 0, zi, epsabs=1e-10, epsrel=1e-10)[0] for zi in z])  # تنظیم برای پایداری integral
    return 5 * np.log10(dl * (1 + z)) + 25  # distance modulus

# گام ۴: توابع Likelihood (همان قبلی)
def log_likelihood_cmb(theta, ell, cl_obs, inv_cov):
    cl_model = model_uqgpf(ell, theta)
    residual = cl_obs - cl_model
    return -0.5 * np.dot(residual, np.dot(inv_cov, residual))

def log_likelihood_snia(theta, z, mu_obs, mu_err):
    mu_model = mu_theory(z, theta)
    chi2 = np.sum(((mu_obs - mu_model) / mu_err)**2)
    return -0.5 * chi2

def log_prior(theta):
    Omega_m, h, gamma, beta, alpha = theta
    if 0 < Omega_m < 1 and 0.5 < h < 1 and 0 < gamma < 1 and 0 < beta < 10 and 1 < alpha < 100:
        return 0.0
    return -np.inf

def log_posterior(theta, *args):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    ll_cmb = log_likelihood_cmb(theta, *args[:3])
    ll_snia = log_likelihood_snia(theta, *args[3:])
    return lp + ll_cmb + ll_snia

# گام ۵: اجرای MCMC
ndim = 5
nwalkers = 32
nsteps = 5000  # برای نتایج واقعی؛ برای تست به 1000 تغییر دهید

pos = [0.3, 0.7, 0.4, 5, 50] + 1e-4 * np.random.randn(nwalkers, ndim)
args = (ell, cl_obs, inv_cov_cmb, z_snia, mu_obs_snia, mu_err_snia)

# اگر pool مشکل داشت، این بلوک را کامنت کنید و خطوط زیر را استفاده کنید:
# sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=args)
# sampler.run_mcmc(pos, nsteps, progress=True)
with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=args, pool=pool)
    sampler.run_mcmc(pos, nsteps, progress=True)

# گام ۶: پردازش خروجی‌ها
samples = sampler.get_chain(discard=200, thin=1, flat=True)  # تنظیم برای samples بیشتر
labels = ["\\Omega_m", "h", "\\gamma", "\\beta", "\\alpha"]

# کرنر پلات سفارشی با پنل ell (مشابه تصویر شما)
fig = corner.corner(samples, labels=labels, quantiles=[0.16, 0.5, 0.84], show_titles=True)
# افزودن پنل فیت به پایین‌راست (جایگزین آخرین پنل خالی)
axes = np.array(fig.axes).reshape((ndim, ndim))
theta_med = np.median(samples, axis=0)
cl_model = model_uqgpf(ell, theta_med)
ax = axes[-1, -1]  # پایین‌راست
ax.plot(ell, cl_obs, label="Data", color="blue")
ax.plot(ell, cl_model, label="UQGPF Fit", color="orange")
ax.set_xlabel("ell")
ax.set_ylabel("C_l")
ax.legend()
fig.savefig("uqgpf_posteriors_v8_7.pdf")

# residuals SNIa
mu_model = mu_theory(z_snia, theta_med)
residuals = mu_obs_snia - mu_model
pd.DataFrame(residuals).to_csv("residuals_v8_7.csv", index=False)

# جدول LaTeX
table = "\\begin{table}\n\\centering\n\\begin{tabular}{cc}\nParameter & Value \\\\\n\\hline\n"
for label, val in zip(labels, theta_med):
    table += f"{label} & {val:.4f} \\\\\n"
table += "\\end{tabular}\n\\end{table}"
with open("latex_table_v8_7.tex", "w") as f:
    f.write(table)

# پلات فیت جداگانه (اختیاری)
plt.figure()
plt.plot(ell, cl_obs, label="Data")
plt.plot(ell, cl_model, label="UQGPF Fit")
plt.xlabel("ell")
plt.ylabel("C_l")
plt.legend()
plt.savefig("uqgpf_fit_v8_7.pdf")

print("Execution complete! Outputs generated.")
