import json
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

# Load data
path = 'affairs.csv'
df = pd.read_csv(path)

# Map columns
# feature2: affairs frequency
# feature6: children (yes/no)

affairs = df['feature2']
children = df['feature6'].astype(str).str.lower()

# Create indicator: 1 if children yes
child_yes = (children == 'yes').astype(int)

# Basic group stats
stats_by_group = df.groupby(child_yes)['feature2'].agg(['count','mean','median','std'])

# Two-sample t-test (Welch)
vals_yes = df.loc[child_yes == 1, 'feature2']
vals_no = df.loc[child_yes == 0, 'feature2']

ttest = stats.ttest_ind(vals_yes, vals_no, equal_var=False, nan_policy='omit')

# Mann-Whitney U
mwu = stats.mannwhitneyu(vals_yes, vals_no, alternative='two-sided')

# Effect size: Cohen's d
mean_yes = vals_yes.mean()
mean_no = vals_no.mean()
std_yes = vals_yes.std(ddof=1)
std_no = vals_no.std(ddof=1)
# pooled SD for Cohen's d
n_yes = vals_yes.shape[0]
n_no = vals_no.shape[0]
pooled_sd = np.sqrt(((n_yes-1)*std_yes**2 + (n_no-1)*std_no**2) / (n_yes + n_no - 2))
cohens_d = (mean_yes - mean_no) / pooled_sd

# Binary outcome: any affair
any_affair = (df['feature2'] > 0).astype(int)

# Logistic regression: any_affair ~ child_yes
X = sm.add_constant(child_yes)
logit_model = sm.Logit(any_affair, X).fit(disp=False)
logit_params = logit_model.params
logit_pvalues = logit_model.pvalues
logit_ci = logit_model.conf_int()

# Also OLS for continuous outcome
ols_model = sm.OLS(df['feature2'], X).fit()

results = {
    'group_stats': stats_by_group.to_dict(),
    'ttest': {'statistic': ttest.statistic, 'pvalue': ttest.pvalue},
    'mannwhitneyu': {'statistic': mwu.statistic, 'pvalue': mwu.pvalue},
    'cohens_d': cohens_d,
    'logit': {
        'params': logit_params.to_dict(),
        'pvalues': logit_pvalues.to_dict(),
        'conf_int': logit_ci.rename(columns={0:'ci_low',1:'ci_high'}).to_dict(),
    },
    'ols': {
        'params': ols_model.params.to_dict(),
        'pvalues': ols_model.pvalues.to_dict(),
        'r2': ols_model.rsquared,
    }
}

print(json.dumps(results, indent=2))
