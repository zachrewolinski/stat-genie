import json
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest

with open('info.json', 'r') as f:
    info = json.load(f)

cols = info['data_desc']['fields']

female_cols = [c['column'] for c in cols if 'female' in c['properties'].get('description','').lower()]
approve_cols = [c['column'] for c in cols if 'accepted' in c['properties'].get('description','').lower()]
denied_cols = [c['column'] for c in cols if 'denied' in c['properties'].get('description','').lower()]

if not female_cols:
    raise SystemExit('No gender (female) column found')
if not approve_cols and not denied_cols:
    raise SystemExit('No approval/denial column found')

female_col = female_cols[0]

df = pd.read_csv('mortgage.csv')

approve_col = None
for col in approve_cols:
    if col in df.columns and df[col].mean() > 0.5:
        approve_col = col
        break

# If no obvious approval column, try to construct from denial
if approve_col is None:
    # pick denial column with mean < 0.5 and invert
    denial_col = None
    for col in denied_cols:
        if col in df.columns and df[col].mean() < 0.5:
            denial_col = col
            break
    if denial_col is None:
        approve_col = approve_cols[0]
        approve_series = df[approve_col]
    else:
        approve_col = f"1 - {denial_col}"
        approve_series = 1 - df[denial_col]
else:
    approve_series = df[approve_col]

# Build contingency table
ct = pd.crosstab(df[female_col], approve_series)

female_yes = ct.loc[1, 1] if 1 in ct.index and 1 in ct.columns else np.nan
female_total = ct.loc[1].sum() if 1 in ct.index else np.nan
male_yes = ct.loc[0, 1] if 0 in ct.index and 1 in ct.columns else np.nan
male_total = ct.loc[0].sum() if 0 in ct.index else np.nan

female_rate = female_yes / female_total
male_rate = male_yes / male_total

chi2, p_chi2, _, _ = stats.chi2_contingency(ct)
count = np.array([female_yes, male_yes])
nobs = np.array([female_total, male_total])
z_stat, p_z = proportions_ztest(count, nobs)

prop_diff = female_rate - male_rate

# Odds ratio (with small correction if needed)
cell = ct.copy().astype(float)
if (cell == 0).any().any():
    cell += 0.5
odds_ratio = (cell.loc[1,1] / cell.loc[1,0]) / (cell.loc[0,1] / cell.loc[0,0])

# Choose strong "No" due to negligible effect and non-significance
response = 5

explanation = (
    f"Using the gender indicator (column '{female_col}', 1=female, 0=male) and the approval outcome "
    f"(approval encoded in '{approve_col}'), there is no evidence that gender affects approval. "
    f"Female approval rate = {female_rate:.4f} ({int(female_yes)}/{int(female_total)}), "
    f"male approval rate = {male_rate:.4f} ({int(male_yes)}/{int(male_total)}); "
    f"difference = {prop_diff:.6f} (about {prop_diff*100:.3f} percentage points). "
    f"A chi-square test of independence gives p={p_chi2:.3f}, and a two-proportion z-test gives p={p_z:.3f}; "
    f"both are far from statistical significance. The odds ratio is {odds_ratio:.4f}, essentially 1. "
    f"Thus the data are consistent with no relationship between gender and mortgage approval."
)

with open('conclusion.txt', 'w') as f:
    json.dump({"response": response, "explanation": explanation}, f)
