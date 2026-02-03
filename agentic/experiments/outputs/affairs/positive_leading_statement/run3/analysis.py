import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.weightstats import ttest_ind

# Load data

df = pd.read_csv('affairs.csv')

# Binary indicator for any affair

df['affair_any'] = (df['affairs'] > 0).astype(int)

# Group summaries

group_means = df.groupby('children')['affairs'].mean()
any_rates = df.groupby('children')['affair_any'].mean()

# Two-sample t-test for mean affairs

affairs_yes = df.loc[df['children'] == 'yes', 'affairs']
affairs_no = df.loc[df['children'] == 'no', 'affairs']
t_stat, p_val, dfree = ttest_ind(affairs_yes, affairs_no, usevar='unequal')

# OLS with controls

ols = smf.ols(
    'affairs ~ C(children) + age + yearsmarried + C(gender) + religiousness + '
    'education + occupation + rating',
    data=df
).fit()

# Logit for any affair with controls

logit = smf.logit(
    'affair_any ~ C(children) + age + yearsmarried + C(gender) + religiousness + '
    'education + occupation + rating',
    data=df
).fit(disp=False)

# Output key results

print('Group mean affairs (children=yes):', group_means.get('yes'))
print('Group mean affairs (children=no):', group_means.get('no'))
print('Group any-affair rate (children=yes):', any_rates.get('yes'))
print('Group any-affair rate (children=no):', any_rates.get('no'))
print('T-test for mean affairs (yes vs no): t=%.3f p=%.3f df=%.1f' % (t_stat, p_val, dfree))

print('\nOLS coefficient for children=yes:')
print('coef=%.4f p=%.4f' % (ols.params.get('C(children)[T.yes]'), ols.pvalues.get('C(children)[T.yes]')))

print('\nLogit coefficient for children=yes:')
print('coef=%.4f p=%.4f' % (logit.params.get('C(children)[T.yes]'), logit.pvalues.get('C(children)[T.yes]')))

