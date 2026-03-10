import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest

# Load data
df = pd.read_csv('mortgage.csv')

# Key columns
base_df = df[['feature2', 'feature14']].replace([np.inf, -np.inf], np.nan).dropna()
female = base_df['feature2']
accept = base_df['feature14']

# Basic rates
rate_female = accept[female == 1].mean()
rate_male = accept[female == 0].mean()
count_female = (female == 1).sum()
count_male = (female == 0).sum()

# Two-proportion z-test for difference in acceptance rates
successes = np.array([
    accept[female == 1].sum(),
    accept[female == 0].sum()
])
ns = np.array([count_female, count_male])
stat, pval = proportions_ztest(successes, ns)

# Logistic regression with controls (excluding feature1, feature11, feature14)
controls = ['feature3','feature4','feature5','feature6','feature7','feature8','feature9','feature10','feature12','feature13']
model_df = df[['feature14', 'feature2'] + controls].copy()
model_df = model_df.replace([np.inf, -np.inf], np.nan).dropna()
X = model_df[['feature2'] + controls]
X = sm.add_constant(X)
y = model_df['feature14']

model = sm.Logit(y, X).fit(disp=False)
coef = model.params['feature2']
se = model.bse['feature2']
pval_logit = model.pvalues['feature2']
OR = np.exp(coef)

# Also compute marginal effect at means for context
mfx = model.get_margeff(at='mean', method='dydx')
mfx_frame = mfx.summary_frame()
mfx_summary = mfx_frame.loc['feature2']
pval_col = next(c for c in mfx_frame.columns if 'P' in c or 'Pr' in c)

results = {
    'rate_female': float(rate_female),
    'rate_male': float(rate_male),
    'rate_diff_female_minus_male': float(rate_female - rate_male),
    'count_female': int(count_female),
    'count_male': int(count_male),
    'ztest_stat': float(stat),
    'ztest_pvalue': float(pval),
    'logit_coef_female': float(coef),
    'logit_se_female': float(se),
    'logit_pvalue_female': float(pval_logit),
    'logit_odds_ratio_female': float(OR),
    'marginal_effect_female': float(mfx_summary['dy/dx']),
    'marginal_effect_se': float(mfx_summary['Std. Err.']),
    'marginal_effect_pvalue': float(mfx_summary[pval_col])
}

print(json.dumps(results, indent=2))
