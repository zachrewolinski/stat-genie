import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf


df = pd.read_csv('teachingratings.csv')

# Focus variables
vars_needed = ['beauty','eval','age','gender','minority','native','tenure','division','credits','students']
missing = df[vars_needed].isna().sum()

# Drop rows with missing in key vars
df_model = df[vars_needed].dropna().copy()

n = len(df_model)

# Pearson correlation
r, p = stats.pearsonr(df_model['beauty'], df_model['eval'])

# Simple OLS
m_simple = smf.ols('eval ~ beauty', data=df_model).fit(cov_type='HC3')

# Controls model
formula = 'eval ~ beauty + age + C(gender) + C(minority) + C(native) + C(tenure) + C(division) + C(credits) + students'
m_ctrl = smf.ols(formula, data=df_model).fit(cov_type='HC3')

beauty_sd = df_model['beauty'].std(ddof=1)
eval_sd = df_model['eval'].std(ddof=1)

# Effect per 1 SD beauty
coef_simple = m_simple.params['beauty']
coef_ctrl = m_ctrl.params['beauty']

# standardized effect in eval SDs
std_eff_simple = (coef_simple * beauty_sd) / eval_sd
std_eff_ctrl = (coef_ctrl * beauty_sd) / eval_sd

out = {
    'n': int(n),
    'missing_counts': missing.to_dict(),
    'pearson_r': float(r),
    'pearson_p': float(p),
    'simple_coef': float(coef_simple),
    'simple_p': float(m_simple.pvalues['beauty']),
    'simple_r2': float(m_simple.rsquared),
    'ctrl_coef': float(coef_ctrl),
    'ctrl_p': float(m_ctrl.pvalues['beauty']),
    'ctrl_r2': float(m_ctrl.rsquared),
    'beauty_sd': float(beauty_sd),
    'eval_sd': float(eval_sd),
    'std_eff_simple': float(std_eff_simple),
    'std_eff_ctrl': float(std_eff_ctrl)
}

print(json.dumps(out, indent=2))
