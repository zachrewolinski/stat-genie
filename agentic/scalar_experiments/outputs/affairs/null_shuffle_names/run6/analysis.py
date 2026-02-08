import pandas as pd
import numpy as np


df = pd.read_csv('affairs.csv')

# Determine likely mapping by inspecting unique values and ranges
summary = {}
for col in df.columns:
    ser = df[col]
    if ser.dtype == object:
        summary[col] = {
            'dtype': 'object',
            'unique': ser.unique()[:10].tolist(),
            'nunique': ser.nunique(),
        }
    else:
        summary[col] = {
            'dtype': 'number',
            'min': float(ser.min()),
            'max': float(ser.max()),
            'nunique': int(ser.nunique()),
            'unique_sample': sorted(ser.unique().tolist())[:10],
            'mean': float(ser.mean()),
        }

print('COLUMN SUMMARY')
for k, v in summary.items():
    print(k, v)

# Assume: affairs count column is 'age' based on unique values
# Assume children binary is 'religiousness' (yes/no)

affairs = df['age']
children = df['religiousness']

# basic checks
print('\nchildren value counts')
print(children.value_counts())

# binary any affairs
any_affair = (affairs > 0).astype(int)

# group stats
stats = df.groupby(children).agg(
    n=('age', 'size'),
    mean_affairs=('age', 'mean'),
    median_affairs=('age', 'median'),
    any_affair_rate=('age', lambda s: (s > 0).mean()),
)
print('\nGROUP STATS (by children)')
print(stats)

# difference in means
mean_no = stats.loc['no', 'mean_affairs'] if 'no' in stats.index else np.nan
mean_yes = stats.loc['yes', 'mean_affairs'] if 'yes' in stats.index else np.nan
print('\nmean_affairs yes - no =', mean_yes - mean_no)

# simple effect size (Cohen d)
# map yes/no to groups
for label in ['yes', 'no']:
    if label not in stats.index:
        raise SystemExit(f'missing {label} in children')

a_yes = affairs[children == 'yes']
a_no = affairs[children == 'no']

def cohen_d(x, y):
    nx = len(x)
    ny = len(y)
    sx = x.std(ddof=1)
    sy = y.std(ddof=1)
    s_pooled = np.sqrt(((nx - 1) * sx**2 + (ny - 1) * sy**2) / (nx + ny - 2))
    return (x.mean() - y.mean()) / s_pooled

print('cohen_d (yes - no):', cohen_d(a_yes, a_no))

# t-test (Welch)
from scipy import stats as st

print('t-test welch (yes vs no):', st.ttest_ind(a_yes, a_no, equal_var=False))

# logistic regression for any affairs controlling for plausible confounders
# identify likely confounders by range:
# occupation: 17.5-57 -> age
# children: 0.125-15 -> years married
# yearsmarried: 9-20 -> education
# rating & affairs: 1-5 -> rate_marriage & religiousness

# We need a mapping guess: use occupation as age, children as years married, yearsmarried as education
# rating and affairs correspond to rate_marriage and religiousness. We can include both.

import statsmodels.api as sm

# assemble dataframe
model_df = pd.DataFrame({
    'any_affair': any_affair,
    'children_yes': (children == 'yes').astype(int),
    'age_years': df['occupation'],
    'years_married': df['children'],
    'education_years': df['yearsmarried'],
    'rating': df['rating'],
    'religiousness_scale': df['affairs'],
})

# drop any missing
model_df = model_df.dropna()

X = model_df[['children_yes', 'age_years', 'years_married', 'education_years', 'rating', 'religiousness_scale']]
X = sm.add_constant(X)
logit = sm.Logit(model_df['any_affair'], X).fit(disp=False)
print('\nLOGIT COEF')
print(logit.params)
print('LOGIT PVALUES')
print(logit.pvalues)

# Also linear regression on affairs count
ols = sm.OLS(affairs, sm.add_constant(model_df[['children_yes', 'age_years', 'years_married', 'education_years', 'rating', 'religiousness_scale']])).fit()
print('\nOLS COEF')
print(ols.params)
print('OLS PVALUES')
print(ols.pvalues)
