import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import chi2_contingency, norm

# Load data
_df = pd.read_csv('mortgage.csv')

gender_col = 'denied_PMI'  # metadata: 1 if female, 0 if male
outcome_denied = 'self_employed'  # metadata: 1 if denied, 0 if accepted

# Basic checks
if gender_col not in _df.columns or outcome_denied not in _df.columns:
    raise ValueError('Expected columns not found')

# Drop rows with missing values in relevant columns
relevant_cols = [gender_col, outcome_denied]
df = _df.dropna(subset=relevant_cols).copy()

# Approval indicator
_df['approved'] = 1 - _df[outcome_denied]
df['approved'] = 1 - df[outcome_denied]

# Group statistics
approval_by_gender = df.groupby(gender_col)['approved'].mean()
counts_by_gender = df.groupby(gender_col)['approved'].count()

# Two-proportion z-test for approval rates
# p1: female (gender==1), p0: male (gender==0)
if 0 in counts_by_gender.index and 1 in counts_by_gender.index:
    n0 = counts_by_gender.loc[0]
    n1 = counts_by_gender.loc[1]
    p0 = approval_by_gender.loc[0]
    p1 = approval_by_gender.loc[1]
    # pooled proportion
    pooled = (p0*n0 + p1*n1) / (n0 + n1)
    se = np.sqrt(pooled*(1-pooled)*(1/n0 + 1/n1))
    z = (p1 - p0) / se if se > 0 else np.nan
    pval_z = 2*(1 - norm.cdf(abs(z))) if se > 0 else np.nan
else:
    n0 = n1 = p0 = p1 = pooled = se = z = pval_z = np.nan

# Chi-square test on approval/denial by gender
ct = pd.crosstab(df[gender_col], df['approved'])
chi2, p_chi2, dof, exp = chi2_contingency(ct)

# Logistic regression: denied ~ gender + controls
# Use all numeric columns except outcome, its complement, and gender
exclude = {gender_col, outcome_denied, 'deny'}  # exclude complement if present
covariates = [c for c in _df.columns if c not in exclude]

# Build design matrix
X = _df[covariates].copy()
y = _df[outcome_denied]

# Drop rows with missing in X or y
mask = X.notna().all(axis=1) & y.notna()
X = X[mask]
y = y[mask]

# Add intercept
X = sm.add_constant(X, has_constant='add')

logit_res = None
logit_error = None
female_coef = None
female_p = None
female_or = None

try:
    model = sm.Logit(y, X)
    logit_res = model.fit(disp=False)
    if gender_col in logit_res.params.index:
        female_coef = logit_res.params[gender_col]
        female_p = logit_res.pvalues[gender_col]
        female_or = float(np.exp(female_coef))
except Exception as e:
    logit_error = str(e)

# Write a concise results summary to a JSON-like text for manual inspection
print('N total', len(_df))
print('N used (group)', len(df))
print('Approval rate by gender (0=male,1=female):', approval_by_gender.to_dict())
print('Counts by gender:', counts_by_gender.to_dict())
print('Two-proportion z-test: z', z, 'p', pval_z)
print('Chi-square: chi2', chi2, 'p', p_chi2)
if logit_res is not None:
    print('Logit female coef (denied as outcome):', female_coef, 'OR', female_or, 'p', female_p)
else:
    print('Logit error:', logit_error)
