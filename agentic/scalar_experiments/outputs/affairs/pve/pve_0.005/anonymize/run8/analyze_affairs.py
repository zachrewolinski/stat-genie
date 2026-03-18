import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import json

# Load data
path = 'affairs.csv'
df = pd.read_csv(path)

# Summary by children
groups = df.groupby('feature6')['feature2']
summary = groups.agg(['count', 'mean', 'median', 'std'])

# Split groups
yes = df.loc[df['feature6'] == 'yes', 'feature2']
no = df.loc[df['feature6'] == 'no', 'feature2']

# Welch t-test
_t_stat, _t_p = stats.ttest_ind(yes, no, equal_var=False, nan_policy='omit')

# Mann-Whitney U (two-sided)
try:
    _u_stat, _u_p = stats.mannwhitneyu(yes, no, alternative='two-sided')
except Exception:
    _u_stat, _u_p = np.nan, np.nan

# Cohen's d (yes - no)

def cohens_d(a, b):
    a = a.dropna()
    b = b.dropna()
    n1, n2 = len(a), len(b)
    s1, s2 = a.std(ddof=1), b.std(ddof=1)
    s_pooled = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    return (a.mean() - b.mean()) / s_pooled

_cohen_d = cohens_d(yes, no)

# Any affair indicator
df['any_affair'] = (df['feature2'] > 0).astype(int)
any_affair = df['any_affair']
prop = df.groupby('feature6')['any_affair'].mean()

# Two-proportion z-test
count_yes = any_affair[df['feature6'] == 'yes'].sum()
count_no = any_affair[df['feature6'] == 'no'].sum()

n_yes = (df['feature6'] == 'yes').sum()
n_no = (df['feature6'] == 'no').sum()

p_pool = (count_yes + count_no) / (n_yes + n_no)
se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_yes + 1 / n_no))
if se > 0:
    z = (count_yes / n_yes - count_no / n_no) / se
    z_p = 2 * (1 - stats.norm.cdf(abs(z)))
else:
    z = np.nan
    z_p = np.nan

# Logistic regression: any_affair ~ children + controls

df_model = df.copy()
df_model['children_yes'] = (df_model['feature6'] == 'yes').astype(int)
df_model['female'] = (df_model['feature3'] == 'female').astype(int)

X = df_model[['children_yes', 'female', 'feature4', 'feature5', 'feature7', 'feature8', 'feature9', 'feature10']]
X = sm.add_constant(X)
y = any_affair

logit = sm.Logit(y, X)
res = logit.fit(disp=False)

# OLS on feature2 with same controls
ols = sm.OLS(df_model['feature2'], X).fit()

output = {
    'summary_by_children': summary.to_dict(),
    't_test': {'t': float(_t_stat), 'p': float(_t_p)},
    'mannwhitney': {'u': float(_u_stat), 'p': float(_u_p)},
    'cohens_d_yes_minus_no': float(_cohen_d),
    'prop_any_affair_by_children': prop.to_dict(),
    'two_prop_z_test': {'z': float(z), 'p': float(z_p)},
    'logit_children_coef': {'coef': float(res.params['children_yes']), 'p': float(res.pvalues['children_yes'])},
    'logit_children_odds_ratio': float(np.exp(res.params['children_yes'])),
    'ols_children_coef': {'coef': float(ols.params['children_yes']), 'p': float(ols.pvalues['children_yes'])}
}

print(json.dumps(output, indent=2))
