import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats


df = pd.read_csv('teachingratings.csv')

# Focus on columns relevant to teaching evaluations
cols = [
    'eval', 'beauty', 'age', 'gender', 'minority', 'credits', 'division', 'native',
    'tenure', 'students', 'allstudents'
]

df_sub = df[cols].copy()

# Drop rows with missing values in these columns
initial_n = len(df_sub)
df_sub = df_sub.dropna()
final_n = len(df_sub)

# Pearson correlation between beauty and eval
corr, corr_p = stats.pearsonr(df_sub['beauty'], df_sub['eval'])

# Simple OLS: eval ~ beauty
model_simple = smf.ols('eval ~ beauty', data=df_sub).fit()

# Multivariate OLS with controls
model_full = smf.ols(
    'eval ~ beauty + age + C(gender) + C(minority) + C(credits) + C(division) + C(native) + C(tenure) + students + allstudents',
    data=df_sub
).fit()

# Effect size: change in eval from 10th to 90th percentile of beauty
p10 = df_sub['beauty'].quantile(0.10)
p90 = df_sub['beauty'].quantile(0.90)

effect_simple = model_simple.params['beauty'] * (p90 - p10)
effect_full = model_full.params['beauty'] * (p90 - p10)

# Standard deviation of eval for context
sd_eval = df_sub['eval'].std()

# Collect key stats
results = {
    'n_initial': initial_n,
    'n_final': final_n,
    'corr': corr,
    'corr_p': corr_p,
    'simple_coef': model_simple.params['beauty'],
    'simple_p': model_simple.pvalues['beauty'],
    'simple_r2': model_simple.rsquared,
    'full_coef': model_full.params['beauty'],
    'full_p': model_full.pvalues['beauty'],
    'full_r2': model_full.rsquared,
    'p10_beauty': p10,
    'p90_beauty': p90,
    'effect_simple': effect_simple,
    'effect_full': effect_full,
    'sd_eval': sd_eval,
}

print(results)
