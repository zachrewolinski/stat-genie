import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

# Load data
path = 'teachingratings.csv'
df = pd.read_csv(path)

# Basic cleaning: drop rows with missing key variables
key_cols = ['beauty', 'eval']
df_clean = df.dropna(subset=key_cols).copy()

# Simple Pearson correlation
corr, corr_p = stats.pearsonr(df_clean['beauty'], df_clean['eval'])

# Simple OLS: eval ~ beauty
model_simple = smf.ols('eval ~ beauty', data=df_clean).fit()

# Multiple OLS with controls
formula_controls = (
    'eval ~ beauty + age + C(gender) + C(minority) + C(credits) + '
    'C(division) + C(native) + C(tenure) + students + allstudents'
)
model_controls = smf.ols(formula_controls, data=df_clean).fit()

# Robust (HC3) and cluster-robust SE by professor ID
model_controls_hc3 = model_controls.get_robustcov_results(cov_type='HC3')
model_controls_cluster = model_controls.get_robustcov_results(
    cov_type='cluster', groups=df_clean['prof']
)

# Effect size per 1 SD beauty
beauty_sd = df_clean['beauty'].std()

# Prepare summary stats for beauty coefficient

def coef_summary(model, label='beauty', names=None):
    # Support both labeled Series and unlabeled numpy arrays (robustcov results)
    if hasattr(model.params, 'index'):
        coef = model.params[label]
        se = model.bse[label]
        p = model.pvalues[label]
        ci_low, ci_high = model.conf_int().loc[label]
    else:
        if names is None:
            names = model.model.exog_names
        idx = names.index(label)
        coef = model.params[idx]
        se = model.bse[idx]
        p = model.pvalues[idx]
        ci_low, ci_high = model.conf_int()[idx]
    return {
        'coef': float(coef),
        'se': float(se),
        'p': float(p),
        'ci_low': float(ci_low),
        'ci_high': float(ci_high),
    }

names = model_controls.model.exog_names
results = {
    'n': int(df_clean.shape[0]),
    'corr': float(corr),
    'corr_p': float(corr_p),
    'beauty_sd': float(beauty_sd),
    'simple': coef_summary(model_simple),
    'controls': coef_summary(model_controls),
    'controls_hc3': coef_summary(model_controls_hc3, names=names),
    'controls_cluster': coef_summary(model_controls_cluster, names=names),
    'r2_simple': float(model_simple.rsquared),
    'r2_controls': float(model_controls.rsquared),
}

with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
