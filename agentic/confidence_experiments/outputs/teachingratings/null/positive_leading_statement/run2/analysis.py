import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('teachingratings.csv')

# Basic stats
beauty = _df['beauty']
eval_scores = _df['eval']

n = len(_df)
beauty_mean = beauty.mean()
beauty_sd = beauty.std(ddof=1)
eval_mean = eval_scores.mean()
eval_sd = eval_scores.std(ddof=1)

# Correlation
corr, corr_p = stats.pearsonr(beauty, eval_scores)

# Unadjusted OLS with clustered SE by professor
m1 = smf.ols('eval ~ beauty', data=_df).fit(cov_type='cluster', cov_kwds={'groups': _df['prof']})

# Adjusted OLS with clustered SE by professor
formula = 'eval ~ beauty + age + C(gender) + C(minority) + C(native) + C(tenure) + C(division) + C(credits) + students + allstudents'
m2 = smf.ols(formula, data=_df).fit(cov_type='cluster', cov_kwds={'groups': _df['prof']})

# Extract coefficients
coef1 = m1.params['beauty']
se1 = m1.bse['beauty']
p1 = m1.pvalues['beauty']

coef2 = m2.params['beauty']
se2 = m2.bse['beauty']
p2 = m2.pvalues['beauty']

# Effect sizes
per_sd_effect_m1 = coef1 * beauty_sd
per_sd_effect_m2 = coef2 * beauty_sd

# 95% CI
ci1 = m1.conf_int().loc['beauty'].tolist()
ci2 = m2.conf_int().loc['beauty'].tolist()

results = {
    'n': n,
    'beauty_mean': beauty_mean,
    'beauty_sd': beauty_sd,
    'eval_mean': eval_mean,
    'eval_sd': eval_sd,
    'corr': corr,
    'corr_p': corr_p,
    'm1_coef': coef1,
    'm1_se': se1,
    'm1_p': p1,
    'm1_ci': ci1,
    'm2_coef': coef2,
    'm2_se': se2,
    'm2_p': p2,
    'm2_ci': ci2,
    'm1_effect_per_sd': per_sd_effect_m1,
    'm2_effect_per_sd': per_sd_effect_m2,
    'm1_r2': m1.rsquared,
    'm2_r2': m2.rsquared,
}

with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
