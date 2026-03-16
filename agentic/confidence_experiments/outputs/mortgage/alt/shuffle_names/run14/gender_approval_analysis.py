import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('mortgage.csv')

# Map columns based on info.json descriptions
# 'denied_PMI' column described as female indicator in metadata
# 'deny' column described as acceptance indicator in metadata
gender_col = 'denied_PMI'   # 1 if female, 0 if male (per metadata)
approval_col = 'deny'       # 1 if approved, 0 if denied (per metadata)

# Drop missing values for gender and approval
sub = _df[[gender_col, approval_col]].dropna()

# Ensure binary values
# (If values are float 0/1)
sub[gender_col] = sub[gender_col].astype(int)
sub[approval_col] = sub[approval_col].astype(int)

# Approval rates by gender
rates = sub.groupby(gender_col)[approval_col].agg(['mean','count','sum'])
print('Approval rates by gender (0=male, 1=female):')
print(rates)

# Two-proportion z-test
male = sub[sub[gender_col]==0][approval_col]
female = sub[sub[gender_col]==1][approval_col]

p0 = male.mean(); p1 = female.mean()
n0 = len(male); n1 = len(female)

p_pool = (male.sum() + female.sum()) / (n0 + n1)
se = np.sqrt(p_pool*(1-p_pool)*(1/n0 + 1/n1)) if n0>0 and n1>0 else np.nan
z = (p1 - p0)/se if se>0 else np.nan
pval = 2*(1-stats.norm.cdf(abs(z))) if np.isfinite(z) else np.nan

print('\nTwo-proportion z-test:')
print(f"male approval rate={p0:.4f} (n={n0}), female approval rate={p1:.4f} (n={n1})")
print(f"z={z:.3f}, p-value={pval:.4g}")

# Logistic regression: approval ~ gender
model_simple = smf.logit(f"{approval_col} ~ {gender_col}", data=sub).fit(disp=False)
print('\nLogit approval ~ gender:')
print(model_simple.summary().tables[1])

# Logistic regression with controls (excluding outcome, gender, and high-unique index-like column)
# Identify likely index-like column (almost unique)
other_cols = [c for c in _df.columns if c not in [gender_col, approval_col]]
# exclude high-unique columns
high_unique = [c for c in other_cols if _df[c].nunique() > 0.95*len(_df)]

control_cols = [c for c in other_cols if c not in high_unique]

# Build dataset for controls
control_df = _df[[approval_col, gender_col] + control_cols].dropna()

# If any control is non-numeric, coerce (but all are numeric)

formula = approval_col + ' ~ ' + gender_col + ' + ' + ' + '.join(control_cols)
try:
    model_controls = smf.logit(formula, data=control_df).fit(disp=False)
    print('\nLogit with controls:')
    print(model_controls.summary().tables[1])
except Exception as e:
    print('\nLogit with controls failed:', e)

# Compute odds ratio and CI for gender (simple model)
coef = model_simple.params[gender_col]
se = model_simple.bse[gender_col]

or_val = np.exp(coef)
ci_low = np.exp(coef - 1.96*se)
ci_high = np.exp(coef + 1.96*se)

print('\nGender odds ratio (female vs male) for approval (simple model):')
print(f"OR={or_val:.3f}, 95% CI [{ci_low:.3f}, {ci_high:.3f}]")
