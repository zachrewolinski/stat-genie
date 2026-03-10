import pandas as pd
import statsmodels.formula.api as smf
import numpy as np

path = 'teachingratings.csv'

df = pd.read_csv(path)

# basic regression with robust SE
simple = smf.ols('eval ~ beauty', data=df).fit(cov_type='HC3')

# full model with controls
formula = (
    'eval ~ beauty + age + students + allstudents '
    '+ C(minority) + C(gender) + C(credits) + C(division) + C(native) + C(tenure)'
)
full = smf.ols(formula, data=df).fit(cov_type='HC3')

beauty_sd = df['beauty'].std()
eval_sd = df['eval'].std()

results = {
    'n': int(df.shape[0]),
    'simple_coef': float(simple.params['beauty']),
    'simple_p': float(simple.pvalues['beauty']),
    'simple_ci': [float(x) for x in simple.conf_int().loc['beauty']],
    'full_coef': float(full.params['beauty']),
    'full_p': float(full.pvalues['beauty']),
    'full_ci': [float(x) for x in full.conf_int().loc['beauty']],
    'beauty_sd': float(beauty_sd),
    'eval_sd': float(eval_sd),
    'simple_effect_sd': float(simple.params['beauty'] * beauty_sd),
    'full_effect_sd': float(full.params['beauty'] * beauty_sd),
    'simple_std_effect': float(simple.params['beauty'] * beauty_sd / eval_sd),
    'full_std_effect': float(full.params['beauty'] * beauty_sd / eval_sd),
    'simple_r2': float(simple.rsquared),
    'full_r2': float(full.rsquared),
}

# partial R2 for beauty in full model using t-stat
from math import sqrt

try:
    t = float(full.tvalues['beauty'])
    df_resid = float(full.df_resid)
    partial_r2 = t**2 / (t**2 + df_resid)
    results['full_partial_r2'] = float(partial_r2)
except Exception:
    results['full_partial_r2'] = None

print(results)
