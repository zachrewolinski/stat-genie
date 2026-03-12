import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

# Load data
_df = pd.read_csv('mortgage.csv')

# Column aliases
female = _df['feature2']
accepted = _df['feature14']

# Basic crosstab
ct = pd.crosstab(female, accepted)
# Ensure columns 0,1
for col in [0,1]:
    if col not in ct.columns:
        ct[col] = 0
ct = ct[[0,1]]

# Rates
rates = (ct[1] / ct.sum(axis=1)).rename('accept_rate')

# Difference in proportions (female - male)
# female==1, male==0
n_f = ct.loc[1].sum() if 1 in ct.index else 0
n_m = ct.loc[0].sum() if 0 in ct.index else 0
p_f = ct.loc[1,1] / n_f if n_f else np.nan
p_m = ct.loc[0,1] / n_m if n_m else np.nan

diff = p_f - p_m
# standard error for diff
se = np.sqrt(p_f*(1-p_f)/n_f + p_m*(1-p_m)/n_m) if n_f and n_m else np.nan
ci_low = diff - 1.96*se if se==se else np.nan
ci_high = diff + 1.96*se if se==se else np.nan

# Chi-square test
chi2, pval, dof, exp = stats.chi2_contingency(ct)

# Logistic regression: acceptance ~ female (unadjusted)
X1 = sm.add_constant(female)
model1 = sm.Logit(accepted, X1, missing='drop')
res1 = model1.fit(disp=False)

# Logistic regression: acceptance ~ female + other covariates (control)
# Select available covariates excluding acceptance/denial
covar_cols = [
    'feature3','feature4','feature5','feature6','feature7','feature8','feature9','feature10','feature12','feature13'
]
X2 = sm.add_constant(_df[covar_cols + ['feature2']])
model2 = sm.Logit(accepted, X2, missing='drop')
res2 = model2.fit(disp=False)

# Extract odds ratios and p-values
or_female_unadj = np.exp(res1.params['feature2'])
ci_unadj = np.exp(res1.conf_int().loc['feature2'].values)

or_female_adj = np.exp(res2.params['feature2'])
ci_adj = np.exp(res2.conf_int().loc['feature2'].values)

out = {
    'counts': ct.to_dict(),
    'accept_rate_female': p_f,
    'accept_rate_male': p_m,
    'diff_female_minus_male': diff,
    'diff_ci_95': [ci_low, ci_high],
    'chi2_pvalue': pval,
    'logit_unadj_coef': res1.params['feature2'],
    'logit_unadj_pvalue': res1.pvalues['feature2'],
    'logit_unadj_or': or_female_unadj,
    'logit_unadj_or_ci': ci_unadj.tolist(),
    'logit_adj_coef': res2.params['feature2'],
    'logit_adj_pvalue': res2.pvalues['feature2'],
    'logit_adj_or': or_female_adj,
    'logit_adj_or_ci': ci_adj.tolist(),
    'n': len(_df),
}

print(out)
