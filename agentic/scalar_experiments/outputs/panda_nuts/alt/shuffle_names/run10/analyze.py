import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# Load data
path = 'panda_nuts.csv'
df = pd.read_csv(path)

# Map shuffled columns to semantic variables based on info.json metadata + data inspection
# age column appears to be chimpanzee ID (1-22), hammer column appears to be age in years (3-16)
# nuts_opened column is sex (m/f), sex column is hammer type, help column is nuts opened count
# chimpanzee column is session duration in seconds, seconds column is received help (y/N)

df = df.rename(columns={
    'age': 'chimpanzee_id',
    'hammer': 'age_years',
    'nuts_opened': 'sex',
    'sex': 'hammer_type',
    'help': 'nuts_opened',
    'chimpanzee': 'seconds',
    'seconds': 'help'
})

# Efficiency: nuts opened per second
# Guard against zero seconds, though not expected

df['efficiency'] = df['nuts_opened'] / df['seconds']

# Basic descriptives
n = len(df)

desc = df[['efficiency', 'age_years', 'nuts_opened', 'seconds']].describe()

# Encode categorical variables
# help: y/N -> 1/0
# sex: m/f -> m as reference

_df = df.copy()
_df['help_binary'] = _df['help'].map({'y': 1, 'N': 0})

# Some rows may have lowercase 'n' or other variations; ensure no missing
if _df['help_binary'].isna().any():
    # Try alternative mapping
    _df['help_binary'] = _df['help'].str.strip().str.lower().map({'y': 1, 'n': 0})

# OLS on efficiency
model = smf.ols('efficiency ~ age_years + C(sex) + help_binary', data=_df).fit(cov_type='HC3')

# Also fit on log1p efficiency to check robustness
_df['log_efficiency'] = np.log1p(_df['efficiency'])
model_log = smf.ols('log_efficiency ~ age_years + C(sex) + help_binary', data=_df).fit(cov_type='HC3')

# Nonparametric / bivariate checks
# Age: Spearman correlation with efficiency
spearman_age = stats.spearmanr(_df['age_years'], _df['efficiency'])

# Sex: Mann-Whitney U (m vs f)
sex_groups = _df.groupby('sex')['efficiency']
if set(_df['sex'].unique()) >= {'m', 'f'}:
    eff_m = _df[_df['sex'] == 'm']['efficiency']
    eff_f = _df[_df['sex'] == 'f']['efficiency']
    mw_sex = stats.mannwhitneyu(eff_m, eff_f, alternative='two-sided')
else:
    mw_sex = None

# Help: Mann-Whitney U (help vs no help)
help_groups = _df.groupby('help_binary')['efficiency']
if set(_df['help_binary'].dropna().unique()) >= {0, 1}:
    eff_help = _df[_df['help_binary'] == 1]['efficiency']
    eff_nohelp = _df[_df['help_binary'] == 0]['efficiency']
    mw_help = stats.mannwhitneyu(eff_help, eff_nohelp, alternative='two-sided')
else:
    mw_help = None

# Summaries for output
results = {
    'n': n,
    'efficiency_mean': df['efficiency'].mean(),
    'efficiency_median': df['efficiency'].median(),
    'ols_params': model.params.to_dict(),
    'ols_pvalues': model.pvalues.to_dict(),
    'ols_r2': model.rsquared,
    'ols_adj_r2': model.rsquared_adj,
    'log_ols_params': model_log.params.to_dict(),
    'log_ols_pvalues': model_log.pvalues.to_dict(),
    'log_ols_r2': model_log.rsquared,
    'log_ols_adj_r2': model_log.rsquared_adj,
    'spearman_age_r': spearman_age.correlation,
    'spearman_age_p': spearman_age.pvalue,
}

if mw_sex:
    results['mw_sex_u'] = mw_sex.statistic
    results['mw_sex_p'] = mw_sex.pvalue

if mw_help:
    results['mw_help_u'] = mw_help.statistic
    results['mw_help_p'] = mw_help.pvalue

# Save results to a json for inspection
import json
with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
