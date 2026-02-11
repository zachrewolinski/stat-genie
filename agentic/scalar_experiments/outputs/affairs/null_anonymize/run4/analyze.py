import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# Load data
path = 'affairs.csv'

df = pd.read_csv(path)

# Clean/ensure types
# feature2 numeric outcome; feature6 children yes/no
# feature3 gender

# Basic group stats
outcome = 'feature2'
child = 'feature6'

# group stats
summary = df.groupby(child)[outcome].agg(['count','mean','median','std'])

# proportion with any affairs
prop_any = df.assign(any_affair=df[outcome] > 0).groupby(child)['any_affair'].mean()

# Welch t-test on means
child_yes = df[df[child]=='yes'][outcome]
child_no = df[df[child]=='no'][outcome]

ttest = stats.ttest_ind(child_yes, child_no, equal_var=False, nan_policy='omit')

# Mann-Whitney U test (two-sided)
try:
    mwu = stats.mannwhitneyu(child_yes, child_no, alternative='two-sided')
except ValueError:
    mwu = None

# OLS regression with controls
formula = (
    "feature2 ~ C(feature6) + C(feature3) + feature4 + feature5 + feature7 + feature8 + feature9 + feature10"
)
ols = smf.ols(formula, data=df).fit(cov_type='HC1')

# Poisson regression (GLM) with robust SE
poisson = smf.glm(formula, data=df, family=sm.families.Poisson()).fit(cov_type='HC1')

results = {
    'summary': summary.to_dict(),
    'prop_any': prop_any.to_dict(),
    'ttest': {
        'statistic': float(ttest.statistic),
        'pvalue': float(ttest.pvalue),
    },
    'mwu': None if mwu is None else {'statistic': float(mwu.statistic), 'pvalue': float(mwu.pvalue)},
    'ols_coef': {
        'coef': float(ols.params.get('C(feature6)[T.yes]', np.nan)),
        'pvalue': float(ols.pvalues.get('C(feature6)[T.yes]', np.nan)),
    },
    'poisson_coef': {
        'coef': float(poisson.params.get('C(feature6)[T.yes]', np.nan)),
        'pvalue': float(poisson.pvalues.get('C(feature6)[T.yes]', np.nan)),
    },
}

with open('analysis_results.json','w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
