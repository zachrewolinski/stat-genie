import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# Load data
_df = pd.read_csv('affairs.csv')

# Normalize column names if needed
# (Assuming they already match info.json)

# Basic checks
n = len(_df)

# Create binary indicator for any affairs
_df['any_affair'] = (_df['affairs'] > 0).astype(int)

# Group summaries by children
summary = _df.groupby('children').agg(
    n=('affairs','size'),
    mean_affairs=('affairs','mean'),
    median_affairs=('affairs','median'),
    any_affair_rate=('any_affair','mean')
)

# Two-sample t-test for mean affairs
children_yes = _df[_df['children']=='yes']['affairs']
children_no = _df[_df['children']=='no']['affairs']

# Welch t-test
_ttest = stats.ttest_ind(children_yes, children_no, equal_var=False, nan_policy='omit')

# Mann-Whitney U
_mw = stats.mannwhitneyu(children_yes, children_no, alternative='two-sided')

# Difference in any-affair rate (proportion)
rate_yes = _df[_df['children']=='yes']['any_affair'].mean()
rate_no = _df[_df['children']=='no']['any_affair'].mean()

# Two-proportion z-test
count_yes = _df[_df['children']=='yes']['any_affair'].sum()
count_no = _df[_df['children']=='no']['any_affair'].sum()

n_yes = _df[_df['children']=='yes'].shape[0]
n_no = _df[_df['children']=='no'].shape[0]

p_pool = (count_yes + count_no) / (n_yes + n_no)
se = np.sqrt(p_pool * (1 - p_pool) * (1/n_yes + 1/n_no))
if se == 0:
    z = np.nan
    p_z = np.nan
else:
    z = (rate_yes - rate_no) / se
    p_z = 2 * (1 - stats.norm.cdf(abs(z)))

# Regression models
# OLS on affairs (not ideal for zero-inflated, but provides directional effect)
ols = smf.ols('affairs ~ C(children) + C(gender) + age + yearsmarried + religiousness + education + occupation + rating', data=_df).fit()

# Logit on any_affair
logit = smf.logit('any_affair ~ C(children) + C(gender) + age + yearsmarried + religiousness + education + occupation + rating', data=_df).fit(disp=False)

# Extract children effect
# Baseline: children=no, coefficient for children=yes
ols_coef = ols.params.get('C(children)[T.yes]', np.nan)
ols_p = ols.pvalues.get('C(children)[T.yes]', np.nan)

logit_coef = logit.params.get('C(children)[T.yes]', np.nan)
logit_p = logit.pvalues.get('C(children)[T.yes]', np.nan)

# Convert logit coef to odds ratio
logit_or = np.exp(logit_coef) if pd.notnull(logit_coef) else np.nan

print('N:', n)
print('\nSummary by children:')
print(summary)
print('\nMean affairs Welch t-test:', _ttest)
print('Mann-Whitney U:', _mw)
print('\nAny-affair rate yes:', rate_yes, 'no:', rate_no)
print('Two-proportion z:', z, 'p:', p_z)
print('\nOLS children coef:', ols_coef, 'p:', ols_p)
print('Logit children coef:', logit_coef, 'OR:', logit_or, 'p:', logit_p)
