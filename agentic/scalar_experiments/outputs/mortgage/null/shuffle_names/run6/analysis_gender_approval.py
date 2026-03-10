import pandas as pd
import numpy as np
from statsmodels.formula.api import logit
from statsmodels.stats.proportion import proportions_ztest

# Load data
_df = pd.read_csv('mortgage.csv')

# Based on info.json descriptions:
# - Column 'denied_PMI' has description "1 if applicant is female, 0 if male" -> use as gender
# - Column 'deny' has description "1 if mortgage application was accepted, 0 if denied" -> use as approval

gender_col = 'denied_PMI'
approval_col = 'deny'

# Drop rows with missing gender or approval
_df = _df[[gender_col, approval_col] + [c for c in _df.columns if c not in [gender_col, approval_col]]].copy()
_df = _df.dropna(subset=[gender_col, approval_col])

# Basic counts
n_total = len(_df)

# Approval rates by gender
rates = _df.groupby(gender_col)[approval_col].agg(['mean','count','sum'])

# Proportion test (two-proportion z-test)
# group0: male (0), group1: female (1)
if set(_df[gender_col].unique()) == {0,1}:
    count = np.array([
        _df[_df[gender_col]==0][approval_col].sum(),
        _df[_df[gender_col]==1][approval_col].sum()
    ])
    nobs = np.array([
        (_df[gender_col]==0).sum(),
        (_df[gender_col]==1).sum()
    ])
    zstat, pval = proportions_ztest(count, nobs)
else:
    zstat, pval = np.nan, np.nan

# Logistic regression: approval ~ female
logit_simple = logit(f"{approval_col} ~ {gender_col}", data=_df).fit(disp=0)

# Logistic regression with controls (exclude obvious ID column)
exclude_cols = {gender_col, approval_col, 'bad_history'}
control_cols = [c for c in _df.columns if c not in exclude_cols]
# keep numeric columns only
control_cols = [c for c in control_cols if pd.api.types.is_numeric_dtype(_df[c])]
# Build formula
formula = f"{approval_col} ~ {gender_col}"
if control_cols:
    formula = f"{approval_col} ~ {gender_col} + " + " + ".join(control_cols)

logit_controls = logit(formula, data=_df).fit(disp=0, maxiter=200)

# Print key results
print("Total observations:", n_total)
print("Approval rate by gender (mean approval):")
print(rates)
print("\nTwo-proportion z-test (approval rate male vs female):")
print("z=", zstat, "p=", pval)
print("\nLogit approval ~ female:")
print(logit_simple.summary())
print("\nLogit approval ~ female + controls:")
print(logit_controls.summary())
