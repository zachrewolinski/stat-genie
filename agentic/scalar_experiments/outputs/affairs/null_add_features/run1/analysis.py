import pandas as pd
import numpy as np
from scipy import stats

# Load data

df = pd.read_csv('affairs.csv')

# Basic cleaning

df = df.copy()

# Ensure children column exists and drop missing

df = df[df['children'].notna() & df['affairs'].notna()]

# Normalize children labels

df['children'] = df['children'].astype(str).str.lower().str.strip()

# Keep only yes/no

df = df[df['children'].isin(['yes', 'no'])]

# Define affair indicator

df['affair_any'] = (df['affairs'] > 0).astype(int)

# Group stats

grp = df.groupby('children')

stats_tbl = grp['affairs'].agg(['count', 'mean', 'median', 'std'])

# Proportion any affair

prop_tbl = grp['affair_any'].mean()

# t-test for mean affairs (Welch)

yes = df[df['children'] == 'yes']['affairs']
no = df[df['children'] == 'no']['affairs']

t_res = stats.ttest_ind(yes, no, equal_var=False, nan_policy='omit')

# Mann-Whitney U (nonparametric)

try:
    u_res = stats.mannwhitneyu(yes, no, alternative='two-sided')
except Exception:
    u_res = None

# Chi-square for any affair

contingency = pd.crosstab(df['children'], df['affair_any'])
chi2_res = stats.chi2_contingency(contingency)

# Effect sizes

mean_diff = yes.mean() - no.mean()
# Cohen's d (pooled std)

pooled_std = np.sqrt(
    ((yes.std(ddof=1) ** 2) * (len(yes) - 1) + (no.std(ddof=1) ** 2) * (len(no) - 1))
    / (len(yes) + len(no) - 2)
)

cohen_d = mean_diff / pooled_std if pooled_std > 0 else np.nan

# Odds ratio for any affair

if 'yes' in contingency.index and 'no' in contingency.index:
    yes_yes = contingency.loc['yes', 1]
    yes_no = contingency.loc['yes', 0]
    no_yes = contingency.loc['no', 1]
    no_no = contingency.loc['no', 0]
    # add 0.5 Haldane-Anscombe if any zero
    if min(yes_yes, yes_no, no_yes, no_no) == 0:
        yes_yes += 0.5
        yes_no += 0.5
        no_yes += 0.5
        no_no += 0.5
    odds_ratio = (yes_yes / yes_no) / (no_yes / no_no)
else:
    odds_ratio = np.nan

print('N:', len(df))
print('Children counts:\n', df['children'].value_counts())
print('Affairs stats by children:\n', stats_tbl)
print('Proportion any affair by children:\n', prop_tbl)
print('Mean diff (yes-no):', mean_diff)
print('Cohen d:', cohen_d)
print('t-test:', t_res)
print('Mann-Whitney:', u_res)
print('Chi-square:', chi2_res)
print('Odds ratio (yes vs no):', odds_ratio)
