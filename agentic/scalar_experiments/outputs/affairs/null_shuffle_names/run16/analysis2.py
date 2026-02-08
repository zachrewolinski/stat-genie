import pandas as pd
import numpy as np
from scipy import stats


df = pd.read_csv('affairs.csv')

# Map columns based on value patterns
# affairs count is in column with values {0,1,2,3,7,12}
affairs_col = None
for col in df.columns:
    vals = sorted(df[col].dropna().unique())
    if len(vals) == 6 and vals == [0,1,2,3,7,12]:
        affairs_col = col
        break
if affairs_col is None:
    raise ValueError('affairs column not found')

# children is yes/no column
children_col = None
for col in df.columns:
    if df[col].dtype == object:
        vals = sorted(df[col].dropna().unique())
        if vals == ['no','yes']:
            # possible children or gender. Determine which by checking other column
            # gender has values male/female
            if col != 'gender':
                children_col = col
                break

if children_col is None:
    # maybe children encoded as 0/1
    for col in df.columns:
        vals = sorted(df[col].dropna().unique())
        if vals == [0,1]:
            children_col = col
            break

if children_col is None:
    raise ValueError('children column not found')

# Prepare data

df['has_children'] = df[children_col].map({'yes':1,'no':0}) if df[children_col].dtype == object else df[children_col].astype(int)
df['affairs'] = df[affairs_col].astype(float)

groups = df.groupby('has_children')['affairs']
summary = groups.agg(['count','mean','median'])
summary['prop_any_affair'] = groups.apply(lambda s: (s>0).mean())

print('affairs_col', affairs_col)
print('children_col', children_col)
print(summary)

# Effect size for mean difference (Cohen's d)
no = df.loc[df['has_children']==0,'affairs']
yes = df.loc[df['has_children']==1,'affairs']

diff = yes.mean() - no.mean()
# pooled std
pooled_std = np.sqrt(((no.var(ddof=1)) + (yes.var(ddof=1))) / 2)
cohen_d = diff / pooled_std if pooled_std else np.nan

# t-test (Welch)
t_stat, t_p = stats.ttest_ind(yes, no, equal_var=False)

# Mann-Whitney U (nonparam)
try:
    u_stat, u_p = stats.mannwhitneyu(yes, no, alternative='two-sided')
except ValueError:
    u_stat, u_p = np.nan, np.nan

# chi-square for any affair
cont = pd.crosstab(df['has_children'], df['affairs']>0)
chi2, chi_p, _, _ = stats.chi2_contingency(cont)

print('mean diff yes-no:', diff)
print('cohen_d:', cohen_d)
print('t_p:', t_p)
print('mannwhitney_p:', u_p)
print('chi2_p (any affair):', chi_p)
print('counts any affair', cont)
