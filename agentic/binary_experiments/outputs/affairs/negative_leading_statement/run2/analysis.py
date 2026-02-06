import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.weightstats import ttest_ind

# Load data
path = 'affairs.csv'
df = pd.read_csv(path)

# Derived variable: any affairs
DF_AFFAIR_ANY = (df['affairs'] > 0).astype(int)
df = df.assign(affair_any=DF_AFFAIR_ANY)

# Descriptive stats by children
group_stats = df.groupby('children').agg(
    n=('affairs', 'size'),
    mean_affairs=('affairs', 'mean'),
    median_affairs=('affairs', 'median'),
    prop_any=('affair_any', 'mean')
)

# Two-sample t-test for mean affairs (children yes vs no)
children_yes = df.loc[df['children'] == 'yes', 'affairs']
children_no = df.loc[df['children'] == 'no', 'affairs']
t_stat, p_val, dfree = ttest_ind(children_yes, children_no, usevar='unequal')

# OLS with controls (robust SE)
ols = smf.ols(
    'affairs ~ C(children) + C(gender) + age + yearsmarried + religiousness + '
    'education + occupation + rating',
    data=df
).fit(cov_type='HC3')

# Logistic regression for any affairs (odds)
logit = smf.logit(
    'affair_any ~ C(children) + C(gender) + age + yearsmarried + religiousness + '
    'education + occupation + rating',
    data=df
).fit(disp=0)

# Poisson regression for count outcome (robust SE)
poisson = smf.glm(
    'affairs ~ C(children) + C(gender) + age + yearsmarried + religiousness + '
    'education + occupation + rating',
    data=df,
    family=sm.families.Poisson()
).fit(cov_type='HC0')

# Extract key coefficient for children
coef_label = 'C(children)[T.yes]'
ols_coef = ols.params[coef_label]
ols_p = ols.pvalues[coef_label]
logit_coef = logit.params[coef_label]
logit_p = logit.pvalues[coef_label]
poisson_coef = poisson.params[coef_label]
poisson_p = poisson.pvalues[coef_label]

# Odds ratio for logit
odds_ratio = float(np.exp(logit_coef))

print('Descriptive statistics by children:')
print(group_stats)
print('\nDifference in mean affairs (yes - no):')
print(f"t={t_stat:.3f}, p={p_val:.3f}, df={dfree:.1f}")

print('\nOLS (affairs count) coefficient for children=yes:')
print(f"coef={ols_coef:.3f}, p={ols_p:.3f}")

print('\nLogit (any affair) coefficient for children=yes:')
print(f"coef={logit_coef:.3f}, odds_ratio={odds_ratio:.3f}, p={logit_p:.3f}")

print('\nPoisson (affairs count) coefficient for children=yes:')
print(f"coef={poisson_coef:.3f}, p={poisson_p:.3f}")
