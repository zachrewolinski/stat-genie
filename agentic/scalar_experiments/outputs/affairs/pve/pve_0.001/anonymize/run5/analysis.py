import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('affairs.csv')

# Basic cleaning: ensure feature6 is lower-case
children = df['feature6'].astype(str).str.lower()
# outcome
outcome = df['feature2']

# Groups
mask_yes = children == 'yes'
mask_no = children == 'no'

# Descriptives
mean_yes = outcome[mask_yes].mean()
mean_no = outcome[mask_no].mean()
median_yes = outcome[mask_yes].median()
median_no = outcome[mask_no].median()

n_yes = mask_yes.sum()
n_no = mask_no.sum()

# Welch t-test
welch = stats.ttest_ind(outcome[mask_yes], outcome[mask_no], equal_var=False, nan_policy='omit')

# Mann-Whitney U test (nonparametric)
try:
    mw = stats.mannwhitneyu(outcome[mask_yes], outcome[mask_no], alternative='two-sided')
except ValueError:
    mw = None

# Effect size (Cohen's d)
# Pooled SD for unequal sizes
s1 = outcome[mask_yes].std(ddof=1)
s2 = outcome[mask_no].std(ddof=1)
pooled_sd = np.sqrt(((n_yes - 1) * s1**2 + (n_no - 1) * s2**2) / (n_yes + n_no - 2))
cohens_d = (mean_yes - mean_no) / pooled_sd if pooled_sd and pooled_sd > 0 else np.nan

# Also check proportion with any affair > 0
any_affair = outcome > 0
prop_yes = any_affair[mask_yes].mean()
prop_no = any_affair[mask_no].mean()

# Chi-square / proportion test
cont_table = pd.crosstab(children, any_affair)
chi2 = stats.chi2_contingency(cont_table)

# Simple regression: outcome ~ children
# Encode children (yes=1, no=0)
df = df.copy()
df['children_yes'] = (children == 'yes').astype(int)
ols = smf.ols('feature2 ~ children_yes', data=df).fit()

result = {
    'n_yes': int(n_yes),
    'n_no': int(n_no),
    'mean_yes': float(mean_yes),
    'mean_no': float(mean_no),
    'median_yes': float(median_yes),
    'median_no': float(median_no),
    'welch_t': float(welch.statistic),
    'welch_p': float(welch.pvalue),
    'mw_u': float(mw.statistic) if mw is not None else None,
    'mw_p': float(mw.pvalue) if mw is not None else None,
    'cohens_d': float(cohens_d),
    'prop_any_yes': float(prop_yes),
    'prop_any_no': float(prop_no),
    'chi2': float(chi2[0]),
    'chi2_p': float(chi2[1]),
    'ols_coef': float(ols.params['children_yes']),
    'ols_p': float(ols.pvalues['children_yes']),
}

print(result)
