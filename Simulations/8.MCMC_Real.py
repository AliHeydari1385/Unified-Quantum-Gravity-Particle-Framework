import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# فرض: داده‌های MCMC قبلی به صورت آرایه numpy یا dataframe داریم
# اینجا نمونه مصنوعی می‌سازیم چون corner اجرا نشده بود
# در عمل باید این را با خروجی run_mcmc جایگزین کنی
np.random.seed(42)
samples = pd.DataFrame({
    'lambda_param': np.random.normal(1e-42, 1e-43, 15000),
    'other_param': np.random.normal(5.0, 0.5, 15000)
})

# Trace plot
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axes[0].plot(samples['lambda_param'], alpha=0.6)
axes[0].set_ylabel('lambda_param')
axes[1].plot(samples['other_param'], alpha=0.6, color='orange')
axes[1].set_ylabel('other_param')
axes[1].set_xlabel('Iteration')
plt.tight_layout()
trace_path = '/mnt/data/mcmc_trace.png'
plt.savefig(trace_path)
plt.close()

# Posterior histograms
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].hist(samples['lambda_param'], bins=40, color='skyblue', edgecolor='black')
axes[0].set_title('Posterior of lambda_param')
axes[1].hist(samples['other_param'], bins=40, color='salmon', edgecolor='black')
axes[1].set_title('Posterior of other_param')
plt.tight_layout()
posterior_path = '/mnt/data/mcmc_posteriors.png'
plt.savefig(posterior_path)
plt.close()

# Pair plot ساده
pair_fig = sns.pairplot(samples, diag_kind='hist', plot_kws={'alpha':0.3})
pair_path = '/mnt/data/mcmc_pairs.png'
pair_fig.savefig(pair_path)

trace_path, posterior_path, pair_path