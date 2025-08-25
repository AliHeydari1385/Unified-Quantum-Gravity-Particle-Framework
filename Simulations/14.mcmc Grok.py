import numpy as np
import scipy.integrate as integrate
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
import corner
import time
import urllib.request
from multiprocessing import Pool
import os

# تنظیمات اولیه
SigmaClipping = True  # فعال‌سازی sigma clipping برای شناسایی outliers
TrialNum = 50000000  # تعداد گام‌های MCMC برای اجرای کامل
burn = 5000000  # دوره burn-in (10% از کل گام‌ها)
printstep = 1000000  # نمایش پیشرفت هر 1 میلیون گام
bins = 50  # تعداد بین‌ها برای پلات گوشه‌ای
max_iterations = 10  # حداکثر تعداد تکرارها برای sigma clipping
IterationNumber = 1  # شماره تکرار اولیه برای sigma clipping

# پارامترهای فیت و محدودیت‌ها
ParamNames = ['h', 'Om', 'gamma']
Labels = [r'$h$', r'$\Omega_{m,0}$', r'$\gamma$']
NumParams = len(ParamNames)
parameter_limits = np.array([
    [0.6, 0.8, 0.02],   # h (ثابت هابل کاهش‌یافته)
    [0.1, 0.5, 0.05],   # Omega_m,0 (چگالی ماده)
    [0.1, 0.3, 0.02]    # gamma (ترم کوانتومی/چرخه‌ای)
])

# پیشوند برای فایل‌های خروجی
Prefix = f'SNIa_UQGPF_cyclic_sc_{IterationNumber}' if SigmaClipping else 'SNIa_UQGPF_cyclic'

# دانلود و بارگذاری ماتریس کوواریانس
cov_url = "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES_STAT%2BSYS.cov"
try:
    urllib.request.urlretrieve(cov_url, "Pantheon+SH0ES_STAT+SYS.cov")
    Cov = np.genfromtxt("Pantheon+SH0ES_STAT+SYS.cov", skip_header=1)
    Cov = Cov.reshape((1701, 1701))  # ماتریس کوواریانس 1701x1701
    Cov_inv = np.linalg.inv(Cov)  # محاسبه معکوس ماتریس کوواریانس
    print("ماتریس کوواریانس با موفقیت بارگذاری و معکوس شد.")
except Exception as e:
    print(f"خطا در بارگذاری ماتریس کوواریانس: {e}")
    Cov_inv = None

# بهینه‌سازی انتگرال با جدول lookup
z_grid = np.linspace(0, 2.5, 1000)
Om_grid = np.linspace(0.1, 0.5, 50)
gamma_grid = np.linspace(0.1, 0.3, 50)
integr_table = np.zeros((len(z_grid), len(Om_grid), len(gamma_grid)))
print("محاسبه جدول lookup برای انتگرال...")
for i in range(len(z_grid)):
    for j in range(len(Om_grid)):
        for k in range(len(gamma_grid)):
            integr_table[i, j, k], _ = integrate.quad(
                lambda x: 1.0 / np.sqrt(Om_grid[j] * (1 + x)**3 + (1 - Om_grid[j]) + gamma_grid[k] * (1 + x)**4), 
                0, z_grid[i]
            )
interp_integr = RegularGridInterpolator((z_grid, Om_grid, gamma_grid), integr_table, method='linear')
def Integrate_fast(z, Om, gamma):
    return interp_integr(np.vstack([z, np.full_like(z, Om), np.full_like(z, gamma)]).T)

# تابع برای محاسبه مدول فاصله مشاهده‌شده (mu_measured)
def Mu_measured(data, alpha=0.14, beta=3.1, gamma_SN=0.05, MB=-19.3, tau=10.0):
    mB = data[:, 3]  # ستون mB
    x1 = data[:, 4]  # ستون x1
    co = data[:, 5]  # ستون c
    m_host = data[:, 6]  # ستون HOST_LOGMASS
    d_bias = data[:, 7]  # ستون biasCor_m_b
    delta_host = np.where(m_host > tau, gamma_SN, 0)
    mu_meas = mB - MB - alpha * x1 + beta * co + delta_host + d_bias
    return mu_meas

# تابع محاسبه مدول فاصله نظری (mu_theory)
def Mu_theory(z, h, integr, c=299792458.0):
    H0 = h * 100.0 * 1000.0 / 3.08568e19  # تبدیل h به km/s/Mpc
    DL = (c / H0) * (1 + z) * integr  # مدول فاصله در Mpc
    mu_th = 5 * np.log10(DL) + 25  # مدول فاصله
    return mu_th

# تابع محاسبه chi-squared
def ChiCalculator(data, params, integr, Cov_inv=None):
    z = data[:, 0]  # ستون zHD
    h, Om, gamma = params
    mu_meas = Mu_measured(data)
    mu_th = Mu_theory(z, h, integr, c=299792458.0)
    delta_mu = mu_meas - mu_th
    if Cov_inv is None:
        sigma = 0.1 * np.ones(len(delta_mu))  # واریانس فرضی در صورت عدم وجود ماتریس
        Chi = np.sum((delta_mu / sigma)**2)
    else:
        delta_mu_len = len(delta_mu)
        Chi = np.dot(delta_mu, np.dot(Cov_inv[:delta_mu_len, :delta_mu_len], delta_mu))
    return Chi

# تابع sigma clipping برای شناسایی outliers
def Sigma_Clipping(data, params, integr, outfiltered, iteration):
    z = data[:, 0]
    mu_meas = Mu_measured(data)
    mu_th = Mu_theory(z, params[0], integr)
    delta_mu = mu_meas - mu_th
    sigma = np.sqrt(np.diag(Cov[:len(delta_mu), :len(delta_mu)])) if Cov_inv is not None else 0.1
    mask = np.abs(delta_mu) > 3 * sigma
    outfiltered[mask] = iteration
    return outfiltered

# الگوریتم MCMC برای یک زنجیره
def MCMC(data, integr_init, Prefix, Cov_inv=None, seed=0):
    np.random.seed(seed)
    global integr
    integr = integr_init
    OutStat = 1e10  # مقدار اولیه chi-squared
    Params = np.random.uniform(parameter_limits[:, 0], parameter_limits[:, 1], NumParams)
    OutM = np.zeros((TrialNum + 1, NumParams + 1))
    OutM[0, :-1] = Params
    OutM[0, -1] = OutStat
    start_time = time.time()

    for t in range(TrialNum):
        TestParams = Params.copy()
        ParamInd = np.random.randint(0, NumParams)
        RandStepParam = np.random.normal(0, parameter_limits[ParamInd, 2])
        TestParams[ParamInd] += RandStepParam
        if TestParams[ParamInd] < parameter_limits[ParamInd, 0] or TestParams[ParamInd] > parameter_limits[ParamInd, 1]:
            TestParams[ParamInd] -= 2 * RandStepParam
        if ParamInd == 1 or ParamInd == 2:
            integr = Integrate_fast(data[:, 0], TestParams[1], TestParams[2])
        TestOutStat = ChiCalculator(data, TestParams, integr, Cov_inv)
        a = np.exp(-0.5 * (TestOutStat - OutStat))
        if np.random.random() < a or TestOutStat < OutStat:
            Params = TestParams
            OutStat = TestOutStat
            if ParamInd == 1 or ParamInd == 2:
                integr = Integrate_fast(data[:, 0], Params[1], Params[2])
        OutM[t + 1, :-1] = Params
        OutM[t + 1, -1] = OutStat
        if (t + 1) % printstep == 0:
            elapsed = time.time() - start_time
            print(f"Chain {seed}, Step {t+1}, Time elapsed: {elapsed:.2f} s, Chi2: {OutStat:.2f}")
    np.savetxt(f'OutMatrix_{Prefix}_chain_{seed}.txt', OutM, delimiter='\t')
    return Params

# تابع برای محاسبه پارامترهای بهینه
def OptPar(Prefix, num_chains=4):
    OutM_all = []
    for seed in range(num_chains):
        OutM = np.genfromtxt(f'OutMatrix_{Prefix}_chain_{seed}.txt')
        OutM = OutM[burn:, :-1]  # حذف burn-in و ستون chi-squared
        OutM_all.append(OutM)
    OutM_all = np.vstack(OutM_all)
    quantiles = [0.16, 0.5, 0.84]
    OptParams = []
    for i in range(NumParams):
        opt = corner.quantile(OutM_all[:, i], quantiles)
        OptParams.append([opt[1], opt[1] - opt[0], opt[2] - opt[1]])
    np.savetxt(f'params_{Prefix}.txt', OptParams, delimiter='\t')
    if not SigmaClipping:
        figure = corner.corner(OutM_all, bins=bins, labels=Labels, quantiles=quantiles, show_titles=True, title_fmt='.4f', title_kwargs={"fontsize": 12})
        figure.savefig(f'{Prefix}_posteriors.pdf')
        plt.close()
    return OptParams

# تابع برای اجرای موازی زنجیره‌ها
def run_chain(seed):
    global Prefix
    data_masked = SNIa_data[outfiltered == 0]
    integr_init = Integrate_fast(data_masked[:, 0], 0.3, 0.2)
    return MCMC(data_masked, integr_init, Prefix, Cov_inv, seed)

# تابع اجرای کلی با sigma clipping
def RunThis(data, Cov_inv=None):
    global IterationNumber, outfiltered, Prefix
    outfiltered = np.zeros(len(data))
    if SigmaClipping:
        for iter in range(max_iterations):
            IterationNumber = iter + 1
            Prefix = f'SNIa_UQGPF_cyclic_sc_{IterationNumber}'
            print(f"Starting Iteration {IterationNumber} for Sigma Clipping")
            with Pool(4) as p:  # 4 زنجیره موازی
                results = p.map(run_chain, range(4))
            OptParams = OptPar(Prefix, num_chains=4)
            outfiltered = Sigma_Clipping(data, OptParams[0], integr_init, outfiltered, IterationNumber)
            num_outfiltered = np.sum(outfiltered > 0)
            print(f"End of Iteration {IterationNumber}, Outfiltered SNe: {num_outfiltered}")
            if num_outfiltered == 0:
                break
        np.savetxt(f'outfiltered_{Prefix}.txt', outfiltered, delimiter='\t')
    else:
        Prefix = 'SNIa_UQGPF_cyclic'
        with Pool(4) as p:
            results = p.map(run_chain, range(4))
        OptPar(Prefix, num_chains=4)

# بارگذاری داده‌ها
try:
    SNIa_data = np.genfromtxt('ES_AND_COVARPantheon%2BSH0ES.dat.txt', usecols=(2, 12, 13, 19, 17, 15, 34, 43))
    print("داده‌های Pantheon با موفقیت بارگذاری شد.")
    print(f"تعداد ابرنواخترها: {len(SNIa_data)}")
except Exception as e:
    print(f"خطا در بارگذاری داده‌ها: {e}")
    print("استفاده از داده‌های mock برای تست.")
    SNIa_data = np.array([
        [0.01, 0.0, 0, 32.5, 0.0, 0.0, 10.0, 0.0],
        [0.1, 0.0, 0, 39.0, 0.0, 0.0, 10.0, 0.0],
        [0.5, 0.0, 0, 43.0, 0.0, 0.0, 10.0, 0.0],
        [1.0, 0.0, 0, 44.5, 0.0, 0.0, 10.0, 0.0],
        [1.5, 0.0, 0, 45.2, 0.0, 0.0, 10.0, 0.0]
    ])

# محاسبه اولیه integr
initial_Om = 0.3
initial_gamma = 0.2
integr_init = Integrate_fast(SNIa_data[:, 0], initial_Om, initial_gamma)

# اجرای کد
if __name__ == "__main__":
    RunThis(SNIa_data, Cov_inv)
