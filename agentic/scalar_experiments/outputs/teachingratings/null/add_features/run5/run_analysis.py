import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats

# Load data

df = pd.read_csv('teachingratings.csv')

# Basic info
n = len(df)

# Pearson correlation between beauty and eval
corr, corr_p = stats.pearsonr(df['beauty'], df['eval'])

# Simple OLS
model_simple = smf.ols('eval ~ beauty', data=df).fit()

# Adjusted OLS with common controls
# Use categorical encodings for factor variables
controls_formula = 'age + C(gender) + C(minority) + C(native) + C(tenure) + C(division) + C(credits) + students + allstudents'
model_adj = smf.ols(f'eval ~ beauty + {controls_formula}', data=df).fit()

# Collect key stats
results = {
    'n': n,
    'corr': corr,
    'corr_p': corr_p,
    'simple_coef': model_simple.params['beauty'],
    'simple_p': model_simple.pvalues['beauty'],
    'simple_ci': model_simple.conf_int().loc['beauty'].tolist(),
    'adj_coef': model_adj.params['beauty'],
    'adj_p': model_adj.pvalues['beauty'],
    'adj_ci': model_adj.conf_int().loc['beauty'].tolist(),
    'adj_r2': model_adj.rsquared,
}

print(results)
