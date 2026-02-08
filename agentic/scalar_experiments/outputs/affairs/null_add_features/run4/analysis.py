import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm


df = pd.read_csv('affairs.csv')

# Clean children values
if df['children'].dtype.name != 'category':
    df['children'] = df['children'].astype(str)

# Standardize to lower-case yes/no
children = df['children'].str.strip().str.lower()

# Use only rows with valid children value
mask = children.isin(['yes', 'no'])
sub = df.loc[mask].copy()
sub['children'] = children[mask]

# Affair measures
sub['any_affair'] = sub['affairs'] > 0

# Group stats
stats_tbl = sub.groupby('children').agg(
    n=('affairs', 'size'),
    mean_affairs=('affairs', 'mean'),
    median_affairs=('affairs', 'median'),
    any_affair_rate=('any_affair', 'mean')
)

# Difference in means
yes = sub.loc[sub['children'] == 'yes', 'affairs']
no = sub.loc[sub['children'] == 'no', 'affairs']

# Welch t-test
ttest = stats.ttest_ind(yes, no, equal_var=False, nan_policy='omit')

# Cohen's d (Hedges g)
ny, nn = yes.size, no.size
mean_diff = yes.mean() - no.mean()
var_y = yes.var(ddof=1)
var_n = no.var(ddof=1)
sp = np.sqrt(((ny-1)*var_y + (nn-1)*var_n) / (ny+nn-2))
cohen_d = mean_diff / sp if sp > 0 else np.nan
# Hedges g correction
J = 1 - (3/(4*(ny+nn)-9))
hedges_g = cohen_d * J

# Difference in proportions (any affair)
py = sub.loc[sub['children']=='yes','any_affair'].mean()
pp = sub.loc[sub['children']=='no','any_affair'].mean()
prop_diff = py - pp
# Two-proportion z-test
count = np.array([
    sub.loc[sub['children']=='yes','any_affair'].sum(),
    sub.loc[sub['children']=='no','any_affair'].sum()
])
obs = np.array([ny, nn])
prop_test = sm.stats.proportions_ztest(count, obs)

# Logistic regression: any_affair ~ children (yes=1)
sub['children_yes'] = (sub['children'] == 'yes').astype(int)
X = sm.add_constant(sub['children_yes'])
logit_model = sm.Logit(sub['any_affair'].astype(int), X)
logit_res = logit_model.fit(disp=False)

# Odds ratio
odds_ratio = np.exp(logit_res.params['children_yes'])

# Poisson regression on count
poisson_model = sm.GLM(sub['affairs'], X, family=sm.families.Poisson())
poisson_res = poisson_model.fit()
rate_ratio = np.exp(poisson_res.params['children_yes'])

print('GROUP STATS')
print(stats_tbl)
print('\nMEAN DIFF (yes - no):', mean_diff)
print('Welch t-test:', ttest)
print('Hedges g:', hedges_g)
print('\nPROP any affair (yes - no):', prop_diff)
print('Prop z-test:', prop_test)
print('Logit odds ratio (yes vs no):', odds_ratio)
print('Logit p-value:', logit_res.pvalues['children_yes'])
print('\nPoisson rate ratio (yes vs no):', rate_ratio)
print('Poisson p-value:', poisson_res.pvalues['children_yes'])
