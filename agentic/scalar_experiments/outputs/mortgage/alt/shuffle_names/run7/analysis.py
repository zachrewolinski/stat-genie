import json
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest

with open('info.json', 'r') as f:
    info = json.load(f)

# Map descriptions to columns
cols = info['data_desc']['fields']

# Identify gender column (female)
female_cols = [c['column'] for c in cols if c['properties'].get('description','').lower().find('female') != -1]
# Identify approval column (accepted)
approve_cols = [c['column'] for c in cols if 'accepted' in c['properties'].get('description','').lower()]
# Identify denial column (denied)
denied_cols = [c['column'] for c in cols if 'denied' in c['properties'].get('description','').lower()]

print('female_cols', female_cols)
print('approve_cols', approve_cols)
print('denied_cols', denied_cols)

# Load data
df = pd.read_csv('mortgage.csv')

# We'll choose approval column with description 'accepted'
if not approve_cols:
    raise SystemExit('No approval column found')

female_col = female_cols[0]

# Prefer column that looks like approval (mean > 0.5)
approve_col = None
for col in approve_cols:
    if col in df.columns and df[col].mean() > 0.5:
        approve_col = col
        break
if approve_col is None:
    approve_col = approve_cols[0]

# Inspect candidate approval/denial columns
for col in sorted(set(approve_cols + denied_cols)):
    if col in df.columns:
        print('col', col, 'mean', df[col].mean(), 'unique', np.sort(df[col].dropna().unique())[:5])

if 'deny' in df.columns and 'self_employed' in df.columns:
    comp_rate = (df['deny'] + df['self_employed'] == 1).mean()
    print('deny+self_employed == 1 rate', comp_rate)

# Check values
print('female unique', df[female_col].dropna().unique())
print('approve_col', approve_col, 'unique', df[approve_col].dropna().unique())

# Ensure binary
# Build contingency table
ct = pd.crosstab(df[female_col], df[approve_col])
print('contingency\n', ct)

# Rates
female_yes = ct.loc[1, 1] if 1 in ct.index and 1 in ct.columns else np.nan
female_total = ct.loc[1].sum() if 1 in ct.index else np.nan
male_yes = ct.loc[0, 1] if 0 in ct.index and 1 in ct.columns else np.nan
male_total = ct.loc[0].sum() if 0 in ct.index else np.nan

print('female_accept_rate', female_yes / female_total)
print('male_accept_rate', male_yes / male_total)

# Chi-square
chi2, p, dof, expected = stats.chi2_contingency(ct)
print('chi2', chi2, 'p', p)

# Two-proportion z-test
count = np.array([female_yes, male_yes])
nobs = np.array([female_total, male_total])
stat, pz = proportions_ztest(count, nobs)
print('z', stat, 'p', pz)

# Effect size: difference in proportions
prop_diff = female_yes / female_total - male_yes / male_total
print('prop_diff', prop_diff)

# Odds ratio
# Add 0.5 correction if any zero
cell = ct.copy().astype(float)
if (cell==0).any().any():
    cell += 0.5
odds_ratio = (cell.loc[1,1] / cell.loc[1,0]) / (cell.loc[0,1] / cell.loc[0,0])
print('odds_ratio', odds_ratio)
