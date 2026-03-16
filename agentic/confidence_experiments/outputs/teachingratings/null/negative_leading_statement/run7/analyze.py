import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats

# Load data
_df = pd.read_csv('teachingratings.csv')

# Basic cleaning: ensure categorical as category type
cat_cols = ['minority','gender','credits','division','native','tenure']
for c in cat_cols:
    if c in _df.columns:
        _df[c] = _df[c].astype('category')

# Primary variables
beauty = _df['beauty']
eval_score = _df['eval']

# Correlations
pearson_r, pearson_p = stats.pearsonr(beauty, eval_score)
spearman_r, spearman_p = stats.spearmanr(beauty, eval_score)

# Model 1: simple OLS
m1 = smf.ols('eval ~ beauty', data=_df).fit()

# Model 2: with controls (avoid both students and allstudents to reduce collinearity)
# Use students (participants), age, gender, minority, credits, division, native, tenure
formula_controls = 'eval ~ beauty + age + students + C(gender) + C(minority) + C(credits) + C(division) + C(native) + C(tenure)'
m2 = smf.ols(formula_controls, data=_df).fit()

# Cluster-robust SE by professor (repeated measures)
m2_cluster = m2.get_robustcov_results(cov_type='cluster', groups=_df['prof'])

# Helper to get coefficient by name from cluster results
exog_names = m2_cluster.model.exog_names
beauty_idx = exog_names.index('beauty')

# Effect sizes
beauty_sd = beauty.std()
# Predicted change in eval for +1 SD beauty
m1_effect_sd = m1.params['beauty'] * beauty_sd
m2_effect_sd = m2.params['beauty'] * beauty_sd
m2c_effect_sd = m2_cluster.params[beauty_idx] * beauty_sd

# Confidence intervals
m2c_ci = m2_cluster.conf_int()

results = {
    'n': int(_df.shape[0]),
    'pearson_r': pearson_r,
    'pearson_p': pearson_p,
    'spearman_r': spearman_r,
    'spearman_p': spearman_p,
    'm1_coef': m1.params['beauty'],
    'm1_p': m1.pvalues['beauty'],
    'm1_ci': m1.conf_int().loc['beauty'].tolist(),
    'm1_r2': m1.rsquared,
    'm2_coef': m2.params['beauty'],
    'm2_p': m2.pvalues['beauty'],
    'm2_ci': m2.conf_int().loc['beauty'].tolist(),
    'm2_r2': m2.rsquared,
    'm2_cluster_coef': m2_cluster.params[beauty_idx],
    'm2_cluster_p': m2_cluster.pvalues[beauty_idx],
    'm2_cluster_ci': m2c_ci[beauty_idx].tolist(),
    'm2_cluster_r2': m2_cluster.rsquared,
    'beauty_sd': beauty_sd,
    'm1_effect_sd': m1_effect_sd,
    'm2_effect_sd': m2_effect_sd,
    'm2_cluster_effect_sd': m2c_effect_sd,
}

with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
