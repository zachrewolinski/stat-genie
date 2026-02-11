import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# Load data
df = pd.read_csv('affairs.csv')

# Map children to binary
df['children_yes'] = (df['feature6'].str.lower() == 'yes').astype(int)

# Summary stats
summary = (
    df.groupby('children_yes')['feature2']
    .agg(['count', 'mean', 'median'])
    .rename(index={0: 'no_children', 1: 'children'})
)

# Two-sample t-test (Welch)
g0 = df.loc[df['children_yes'] == 0, 'feature2']
g1 = df.loc[df['children_yes'] == 1, 'feature2']
ttest = stats.ttest_ind(g0, g1, equal_var=False)

# Mann-Whitney U (nonparametric)
try:
    mw = stats.mannwhitneyu(g0, g1, alternative='two-sided')
except Exception:
    mw = None

# OLS regression with controls
# Use formula for readability
formula = 'feature2 ~ children_yes + C(feature3) + feature4 + feature5 + feature7 + feature8 + feature9 + feature10'
ols = smf.ols(formula, data=df).fit(cov_type='HC3')

# Poisson GLM for count-like outcome
glm_poisson = smf.glm(formula, data=df, family=sm.families.Poisson()).fit(cov_type='HC3')

# Collect results
results = {
    'summary': summary.to_dict(),
    'ttest': {'statistic': float(ttest.statistic), 'pvalue': float(ttest.pvalue)},
    'mannwhitney': None if mw is None else {'statistic': float(mw.statistic), 'pvalue': float(mw.pvalue)},
    'ols_children_coef': float(ols.params['children_yes']),
    'ols_children_pvalue': float(ols.pvalues['children_yes']),
    'glm_poisson_children_coef': float(glm_poisson.params['children_yes']),
    'glm_poisson_children_pvalue': float(glm_poisson.pvalues['children_yes']),
}

print(json.dumps(results, indent=2))
