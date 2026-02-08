import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('amtl.csv')

# Basic cleaning
# Ensure no zero sockets (but dataset min 2 per metadata). Drop rows with missing essential fields.
_df = _df.dropna(subset=['num_amtl', 'sockets', 'age', 'prob_male', 'tooth_class', 'genus']).copy()

# Ensure integer counts
_df['num_amtl'] = _df['num_amtl'].astype(int)
_df['sockets'] = _df['sockets'].astype(int)

# Human indicator
_df['human'] = (_df['genus'] == 'Homo sapiens').astype(int)

# Model: binomial GLM with logit link
# Use proportion endog with frequency weights equal to sockets
_df['amtl_rate'] = _df['num_amtl'] / _df['sockets']

formula = 'amtl_rate ~ human + age + prob_male + C(tooth_class)'
model = smf.glm(
    formula=formula,
    data=_df,
    family=sm.families.Binomial(),
    freq_weights=_df['sockets']
).fit()

# Predicted difference in AMTL rate between humans and non-humans
_df_h = _df.copy()
_df_h['human'] = 1
_df_nh = _df.copy()
_df_nh['human'] = 0

pred_h = model.predict(_df_h)
pred_nh = model.predict(_df_nh)

diff = pred_h - pred_nh

# Summaries
coef_human = model.params['human']
se_human = model.bse['human']
pval_human = model.pvalues['human']

mean_diff = diff.mean()
median_diff = np.median(diff)

# Simple bootstrap CI for mean difference
rng = np.random.default_rng(42)
boot_means = []
idx = np.arange(len(_df))
for _ in range(1000):
    sample_idx = rng.choice(idx, size=len(idx), replace=True)
    boot_means.append(diff.iloc[sample_idx].mean())
boot_means = np.array(boot_means)
ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

print('Rows:', len(_df))
print('Human count:', _df['human'].sum())
print('Non-human count:', (1 - _df['human']).sum())
print('Human coef (log-odds):', coef_human)
print('Human coef SE:', se_human)
print('Human coef p-value:', pval_human)
print('Mean predicted diff (human - nonhuman):', mean_diff)
print('Median predicted diff:', median_diff)
print('Bootstrap 95% CI for mean diff:', (ci_low, ci_high))

# Also compute adjusted predicted rates at mean covariates for interpretability
# Use average age/prob_male and proportions of tooth_class.
# Compute marginal predicted means by averaging predictions across observed covariates.

print('Mean predicted rate humans:', pred_h.mean())
print('Mean predicted rate non-humans:', pred_nh.mean())
