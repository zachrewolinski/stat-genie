import json
import math
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# Load data
csv_path = 'panda_nuts.csv'
df = pd.read_csv(csv_path)

# Rename for clarity
# feature1: individual ID
# feature2: age
# feature3: sex (f/m)
# feature5: nuts opened
# feature6: duration seconds
# feature7: help (y/N)

# Compute efficiency: nuts per second
# Avoid division by zero (duration min > 0 per metadata)
df = df.copy()
df['efficiency'] = df['feature5'] / df['feature6']

# Clean categorical variables
# Sex: f/m
# Help: y/N -> boolean 1/0
# Ensure consistent casing

df['sex'] = df['feature3'].astype(str).str.lower()
df['help'] = df['feature7'].astype(str).str.lower()

# Map help to 1/0
help_map = {'y': 1, 'n': 0}
# Some data uses 'N' for no; handle by lowercasing
# Drop rows with missing/unrecognized categories

df['help_bin'] = df['help'].map(help_map)

# Map sex
sex_map = {'m': 1, 'f': 0}
df['sex_bin'] = df['sex'].map(sex_map)

# Drop rows with missing in key columns
analysis_df = df.dropna(subset=['efficiency', 'feature2', 'sex_bin', 'help_bin', 'feature1']).copy()

# For stability, use log1p efficiency due to skew/zeros
analysis_df['log_eff'] = np.log1p(analysis_df['efficiency'])

# Mixed effects model with random intercept by individual
mixed_model = None
mixed_result = None

try:
    mixed_model = smf.mixedlm(
        'log_eff ~ feature2 + sex_bin + help_bin',
        data=analysis_df,
        groups=analysis_df['feature1']
    )
    mixed_result = mixed_model.fit(reml=False, method='lbfgs')
except Exception as e:
    mixed_result = None

# OLS with cluster-robust SE by individual as fallback/sensitivity
ols_model = smf.ols('log_eff ~ feature2 + sex_bin + help_bin', data=analysis_df)
ols_result = ols_model.fit(cov_type='cluster', cov_kwds={'groups': analysis_df['feature1']})

# Also compute simple correlations and group comparisons for intuition
# Age correlation with efficiency
corr_age_eff = stats.pearsonr(analysis_df['feature2'], analysis_df['efficiency'])

# Sex difference in efficiency (t-test)
sex_groups = analysis_df.groupby('sex_bin')['efficiency']
sex_eff_f = sex_groups.get_group(0)
sex_eff_m = sex_groups.get_group(1)
sex_ttest = stats.ttest_ind(sex_eff_f, sex_eff_m, equal_var=False)

# Help difference in efficiency (t-test)
help_groups = analysis_df.groupby('help_bin')['efficiency']
help_eff_no = help_groups.get_group(0)
help_eff_yes = help_groups.get_group(1)
help_ttest = stats.ttest_ind(help_eff_no, help_eff_yes, equal_var=False)

# Summaries
summary = {
    'n_rows': int(len(analysis_df)),
    'n_individuals': int(analysis_df['feature1'].nunique()),
    'efficiency_mean': float(analysis_df['efficiency'].mean()),
    'efficiency_median': float(analysis_df['efficiency'].median()),
    'corr_age_eff_r': float(corr_age_eff.statistic),
    'corr_age_eff_p': float(corr_age_eff.pvalue),
    'sex_ttest_t': float(sex_ttest.statistic),
    'sex_ttest_p': float(sex_ttest.pvalue),
    'help_ttest_t': float(help_ttest.statistic),
    'help_ttest_p': float(help_ttest.pvalue)
}

if mixed_result is not None:
    mixed_params = mixed_result.params
    mixed_pvalues = mixed_result.pvalues
    summary['mixedlm'] = {
        'params': {k: float(v) for k, v in mixed_params.items()},
        'pvalues': {k: float(v) for k, v in mixed_pvalues.items()},
        'aic': float(mixed_result.aic)
    }
else:
    summary['mixedlm'] = None

summary['ols_cluster'] = {
    'params': {k: float(v) for k, v in ols_result.params.items()},
    'pvalues': {k: float(v) for k, v in ols_result.pvalues.items()},
    'r2': float(ols_result.rsquared)
}

# Save stats to json for inspection
with open('analysis_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
