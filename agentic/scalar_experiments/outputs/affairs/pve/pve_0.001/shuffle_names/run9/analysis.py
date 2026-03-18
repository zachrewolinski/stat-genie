import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

# Load data
df = pd.read_csv('affairs.csv')

# According to info.json descriptions:
# - column 'age' describes frequency of extramarital affairs (outcome)
# - column 'religiousness' describes whether there are children in the marriage (predictor)
# We'll use those mappings.

affairs = df['age']
children = df['religiousness']

# Ensure children is binary yes/no
children_clean = children.str.lower().str.strip()

# Drop missing if any
valid = affairs.notna() & children_clean.notna()

affairs = affairs[valid]
children_clean = children_clean[valid]

# Group stats
stats_by_group = affairs.groupby(children_clean).agg(['count', 'mean', 'std', 'median'])

# Two-sample t-test (Welch)
groups = {
    key: affairs[children_clean == key].astype(float) for key in children_clean.unique()
}
if len(groups) == 2 and all(len(v) > 1 for v in groups.values()):
    keys = list(groups.keys())
    t_res = stats.ttest_ind(groups[keys[0]], groups[keys[1]], equal_var=False, nan_policy='omit')
else:
    t_res = None

# Mann-Whitney U test (nonparametric)
if len(groups) == 2 and all(len(v) > 1 for v in groups.values()):
    keys = list(groups.keys())
    mw_res = stats.mannwhitneyu(groups[keys[0]], groups[keys[1]], alternative='two-sided')
else:
    mw_res = None

# OLS regression: affairs ~ children (binary)
# Encode children: yes=1, no=0
children_binary = children_clean.map({'yes': 1, 'no': 0})
model_df = pd.DataFrame({'affairs': affairs, 'children': children_binary}).dropna()
X = sm.add_constant(model_df['children'])
model = sm.OLS(model_df['affairs'], X).fit()

# Effect size (Cohen's d)
if len(groups) == 2 and all(len(v) > 1 for v in groups.values()):
    keys = list(groups.keys())
    g1 = groups[keys[0]].astype(float)
    g2 = groups[keys[1]].astype(float)
    # pooled std
    n1, n2 = len(g1), len(g2)
    s1, s2 = g1.std(ddof=1), g2.std(ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    d = (g1.mean() - g2.mean()) / pooled if pooled > 0 else np.nan
else:
    d = np.nan

print('Group stats (affairs by children):')
print(stats_by_group)
print('\nT-test (Welch):', t_res)
print('Mann-Whitney U:', mw_res)
print('\nOLS summary:')
print(model.summary())
print('\nCohen d (group1 - group2):', d)
