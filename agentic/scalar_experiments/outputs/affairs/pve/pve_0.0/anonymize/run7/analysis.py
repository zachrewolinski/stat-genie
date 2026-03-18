import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('affairs.csv')

# Basic checks
# feature6: children yes/no

# Group stats

group_stats = df.groupby('feature6')['feature2'].agg(['count','mean','std','median']).reset_index()

# Welch t-test

groups = {
    k: v['feature2'].values for k, v in df.groupby('feature6')
}

if 'yes' in groups and 'no' in groups:
    t_stat, p_val = stats.ttest_ind(groups['yes'], groups['no'], equal_var=False, nan_policy='omit')
    # effect size: Cohen's d (Welch)
    mean_yes = np.nanmean(groups['yes'])
    mean_no = np.nanmean(groups['no'])
    sd_yes = np.nanstd(groups['yes'], ddof=1)
    sd_no = np.nanstd(groups['no'], ddof=1)
    n_yes = np.isfinite(groups['yes']).sum()
    n_no = np.isfinite(groups['no']).sum()
    # pooled SD for d
    pooled_sd = np.sqrt(((n_yes-1)*sd_yes**2 + (n_no-1)*sd_no**2) / (n_yes + n_no - 2))
    cohen_d = (mean_yes - mean_no) / pooled_sd if pooled_sd > 0 else np.nan
else:
    t_stat = p_val = cohen_d = np.nan
    mean_yes = mean_no = np.nan
    n_yes = n_no = 0

# Mann-Whitney U test (two-sided)
if 'yes' in groups and 'no' in groups:
    try:
        u_stat, u_p = stats.mannwhitneyu(groups['yes'], groups['no'], alternative='two-sided')
    except ValueError:
        u_stat, u_p = np.nan, np.nan
else:
    u_stat = u_p = np.nan

# Regression with controls
# Controls from info.json: feature3 gender, feature4 age, feature5 years married,
# feature7 religiousness, feature8 education, feature9 occupation, feature10 marriage rating

model = smf.ols('feature2 ~ C(feature6) + C(feature3) + feature4 + feature5 + feature7 + feature8 + feature9 + feature10', data=df).fit(cov_type='HC3')

# Extract coefficient for children yes (baseline is no)
coef_key = 'C(feature6)[T.yes]'
coef = model.params.get(coef_key, np.nan)
coef_se = model.bse.get(coef_key, np.nan)
coef_p = model.pvalues.get(coef_key, np.nan)

# Build results
results = {
    'group_stats': group_stats.to_dict(orient='records'),
    'welch_t': {'t': t_stat, 'p': p_val, 'cohen_d': cohen_d, 'mean_yes': mean_yes, 'mean_no': mean_no, 'n_yes': int(n_yes), 'n_no': int(n_no)},
    'mann_whitney': {'u': u_stat, 'p': u_p},
    'regression': {'coef_yes': coef, 'se': coef_se, 'p': coef_p, 'r2': model.rsquared, 'n': int(model.nobs)}
}

with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
