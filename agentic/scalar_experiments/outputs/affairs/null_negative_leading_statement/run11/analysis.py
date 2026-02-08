import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('affairs.csv')

# Basic cleanup
# Ensure children is categorical with yes/no

df['children'] = df['children'].astype('category')

# Create binary outcome: any affairs > 0

df['any_affair'] = (df['affairs'] > 0).astype(int)

# Summary stats
summary = df.groupby('children').agg(
    n=('affairs', 'size'),
    mean_affairs=('affairs', 'mean'),
    median_affairs=('affairs', 'median'),
    any_affair_rate=('any_affair', 'mean')
)

print('Summary by children:')
print(summary)

# Two-sample t-test on affairs mean (Welch)
from scipy import stats

aff_yes = df.loc[df['children'] == 'yes', 'affairs']
aff_no = df.loc[df['children'] == 'no', 'affairs']

t_stat, p_val = stats.ttest_ind(aff_yes, aff_no, equal_var=False)
print('\nWelch t-test (affairs mean) yes vs no:')
print('t_stat=', t_stat, 'p_val=', p_val)

# Proportion test for any affair rate
count = np.array([
    df.loc[df['children'] == 'yes', 'any_affair'].sum(),
    df.loc[df['children'] == 'no', 'any_affair'].sum()
])
obs = np.array([
    (df['children'] == 'yes').sum(),
    (df['children'] == 'no').sum()
])

from statsmodels.stats.proportion import proportions_ztest

z_stat, p_prop = proportions_ztest(count, obs)
print('\nProportion z-test (any affair rate) yes vs no:')
print('z_stat=', z_stat, 'p_val=', p_prop)

# Regression controls: OLS on affairs (count-ish; still), include children + controls
# We'll use OLS for simple interpretability

model_ols = smf.ols('affairs ~ C(children) + age + yearsmarried + religiousness + education + occupation + rating + C(gender)', data=df).fit()
print('\nOLS regression summary (affairs):')
print(model_ols.summary())

# Logistic regression on any affair
model_logit = smf.logit('any_affair ~ C(children) + age + yearsmarried + religiousness + education + occupation + rating + C(gender)', data=df).fit(disp=False)
print('\nLogit regression summary (any_affair):')
print(model_logit.summary())
