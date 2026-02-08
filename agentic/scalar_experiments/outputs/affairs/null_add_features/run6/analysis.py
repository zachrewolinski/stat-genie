import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = 'affairs.csv'
df = pd.read_csv(path)

# Normalize children column
# Expect values yes/no
children = df['children'].astype(str).str.strip().str.lower()

df = df.copy()
df['children_norm'] = children

# Basic counts
counts = df['children_norm'].value_counts(dropna=False)

# Outcome: affairs count
# Also binary indicator of any affair

df['any_affair'] = (df['affairs'] > 0).astype(int)

# Group summaries
summary = df.groupby('children_norm').agg(
    n=('affairs', 'size'),
    mean_affairs=('affairs', 'mean'),
    median_affairs=('affairs', 'median'),
    prop_any_affair=('any_affair', 'mean')
)

# t-test for difference in mean affairs
kids_yes = df.loc[df['children_norm'] == 'yes', 'affairs']
kids_no = df.loc[df['children_norm'] == 'no', 'affairs']

# Welch t-test
welch_t = stats.ttest_ind(kids_yes, kids_no, equal_var=False, nan_policy='omit')

# Mann-Whitney U test (nonparam)
try:
    mw = stats.mannwhitneyu(kids_yes, kids_no, alternative='two-sided')
except Exception as e:
    mw = e

# Difference in proportions for any affair
p_yes = df.loc[df['children_norm'] == 'yes', 'any_affair'].mean()
p_no = df.loc[df['children_norm'] == 'no', 'any_affair'].mean()

# Standard error for diff in proportions
n_yes = (df['children_norm'] == 'yes').sum()
n_no = (df['children_norm'] == 'no').sum()
se_diff = np.sqrt(p_yes*(1-p_yes)/n_yes + p_no*(1-p_no)/n_no)
prop_diff = p_yes - p_no
z = prop_diff / se_diff if se_diff > 0 else np.nan
p_value_prop = 2*(1-stats.norm.cdf(abs(z))) if np.isfinite(z) else np.nan

# Effect size: Cohen's d
mean_yes = kids_yes.mean()
mean_no = kids_no.mean()
var_yes = kids_yes.var(ddof=1)
var_no = kids_no.var(ddof=1)
# pooled SD for d (use unequal n)
pooled_sd = np.sqrt(((n_yes-1)*var_yes + (n_no-1)*var_no) / (n_yes + n_no - 2))
cohen_d = (mean_yes - mean_no) / pooled_sd if pooled_sd > 0 else np.nan

# Regression controls: OLS on affairs (count), and logit for any affair
# Use numeric covariates available
covars = ['age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating']
# Ensure numeric
for c in covars:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# OLS
ols_formula = 'affairs ~ C(children_norm) + ' + ' + '.join(covars)
ols_model = smf.ols(ols_formula, data=df).fit()

# Logit
logit_formula = 'any_affair ~ C(children_norm) + ' + ' + '.join(covars)
logit_model = smf.logit(logit_formula, data=df).fit(disp=False)

# Print results
print('Counts (children):')
print(counts)
print('\nSummary by children:')
print(summary)
print('\nWelch t-test: stat=%.4f p=%.4g' % (welch_t.statistic, welch_t.pvalue))
print('Mann-Whitney U:', mw)
print('\nAny affair proportions: yes=%.4f no=%.4f diff=%.4f z=%.4f p=%.4g' % (p_yes, p_no, prop_diff, z, p_value_prop))
print('Cohen d (yes-no): %.4f' % cohen_d)

print('\nOLS coeff for children (yes vs no):')
print(ols_model.params.filter(like='C(children_norm)'))
print(ols_model.pvalues.filter(like='C(children_norm)'))

print('\nLogit coeff for children (yes vs no):')
print(logit_model.params.filter(like='C(children_norm)'))
print(logit_model.pvalues.filter(like='C(children_norm)'))

