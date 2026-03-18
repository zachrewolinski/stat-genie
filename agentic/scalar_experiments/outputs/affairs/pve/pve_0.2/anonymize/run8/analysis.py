import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats


df = pd.read_csv('affairs.csv')

# Identify columns
children_col = 'feature6'
affairs_col = 'feature2'

# Clean
# Ensure children is categorical yes/no
children = df[children_col].astype(str).str.lower().str.strip()

# Outcome numeric
affairs = pd.to_numeric(df[affairs_col], errors='coerce')

# Basic group stats
summary = df.groupby(children_col)[affairs_col].agg(['count','mean','median','std'])

# Two-sample t-test (Welch)
no_aff = affairs[children == 'no']
yes_aff = affairs[children == 'yes']

ttest = stats.ttest_ind(no_aff, yes_aff, equal_var=False, nan_policy='omit')

# Mann-Whitney U
mw = stats.mannwhitneyu(no_aff, yes_aff, alternative='two-sided')

# Effect size (Cohen's d)
mean_diff = no_aff.mean() - yes_aff.mean()
# pooled std
n1, n2 = no_aff.shape[0], yes_aff.shape[0]
std1, std2 = no_aff.std(ddof=1), yes_aff.std(ddof=1)
pooled = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2))
cohens_d = mean_diff / pooled if pooled > 0 else np.nan

# Binary outcome: any affairs
any_aff = (affairs > 0).astype(int)
df = df.copy()
df['any_aff'] = any_aff

# Logistic regression: any_aff ~ children (yes/no)
# Use treatment coding: 'no' as reference
logit = smf.logit('any_aff ~ C(feature6)', data=df).fit(disp=False)

# OLS on frequency (robust SE)
ols = smf.ols('feature2 ~ C(feature6)', data=df).fit(cov_type='HC3')

results = {
    'summary': summary.to_dict(),
    'ttest': {'stat': float(ttest.statistic), 'pvalue': float(ttest.pvalue)},
    'mannwhitney': {'stat': float(mw.statistic), 'pvalue': float(mw.pvalue)},
    'mean_diff_no_minus_yes': float(mean_diff),
    'cohens_d': float(cohens_d),
    'logit_params': logit.params.to_dict(),
    'logit_pvalues': logit.pvalues.to_dict(),
    'logit_odds_ratio_children_yes': float(np.exp(logit.params.get('C(feature6)[T.yes]', np.nan))),
    'ols_params': ols.params.to_dict(),
    'ols_pvalues': ols.pvalues.to_dict(),
}

print(json.dumps(results, indent=2))
