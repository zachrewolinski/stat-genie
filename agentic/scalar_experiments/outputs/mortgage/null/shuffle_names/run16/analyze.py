import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Load data

df = pd.read_csv('mortgage.csv')

# Basic checks

n_total = len(df)

# Determine relationship between deny and accept
accept_counts = df['accept'].value_counts(dropna=False).to_dict()
deny_counts = df['deny'].value_counts(dropna=False).to_dict()

# Cross-tab accept vs deny
ct_accept_deny = pd.crosstab(df['accept'], df['deny'])

# Gender vs deny
ct_female_deny = pd.crosstab(df['female'], df['deny'])

# Compute denial rate by gender (assuming deny==1 means denied)

denial_rate_by_gender = (df.groupby('female')['deny'].mean()).to_dict()

# Chi-square test for independence
chi2, p_chi, dof, expected = stats.chi2_contingency(ct_female_deny)

# Logistic regression: deny ~ female + controls
# Define controls as all other columns except deny
control_cols = [c for c in df.columns if c not in ['deny']]

X = df[control_cols].copy()

# Add intercept
X = sm.add_constant(X, has_constant='add')

# Drop rows with missing or infinite values
mask_full = np.isfinite(X).all(axis=1) & np.isfinite(df['deny'])
X_full = X.loc[mask_full]
y_full = df.loc[mask_full, 'deny']

# Logistic regression with robust standard errors
try:
    logit_model = sm.Logit(y_full, X_full)
    # Fit with robust covariance directly if supported
    logit_res = logit_model.fit(disp=False, cov_type='HC1')
    coef_female = logit_res.params.get('female', np.nan)
    se_female = logit_res.bse.get('female', np.nan)
    p_female = logit_res.pvalues.get('female', np.nan)
    odds_ratio_female = np.exp(coef_female) if pd.notnull(coef_female) else np.nan
except Exception as e:
    logit_res = None
    coef_female = se_female = p_female = odds_ratio_female = np.nan
    logit_error = str(e)
else:
    logit_error = None

# Also simple logistic regression with only female
X_simple = sm.add_constant(df[['female']], has_constant='add')
mask_simple = np.isfinite(X_simple).all(axis=1) & np.isfinite(df['deny'])
X_simple = X_simple.loc[mask_simple]
y_simple = df.loc[mask_simple, 'deny']
try:
    logit_simple = sm.Logit(y_simple, X_simple)
    logit_simple_res = logit_simple.fit(disp=False, cov_type='HC1')
    coef_female_simple = logit_simple_res.params.get('female', np.nan)
    p_female_simple = logit_simple_res.pvalues.get('female', np.nan)
    or_female_simple = np.exp(coef_female_simple) if pd.notnull(coef_female_simple) else np.nan
except Exception as e:
    coef_female_simple = p_female_simple = or_female_simple = np.nan
    logit_simple_error = str(e)
else:
    logit_simple_error = None

output = {
    "n_total": n_total,
    "accept_counts": accept_counts,
    "deny_counts": deny_counts,
    "ct_accept_deny": ct_accept_deny.to_dict(),
    "ct_female_deny": ct_female_deny.to_dict(),
    "denial_rate_by_gender": denial_rate_by_gender,
    "chi2": chi2,
    "chi2_p": p_chi,
    "logit_female_coef": coef_female,
    "logit_female_se": se_female,
    "logit_female_p": p_female,
    "logit_female_or": odds_ratio_female,
    "logit_error": logit_error,
    "logit_simple_female_coef": coef_female_simple,
    "logit_simple_female_p": p_female_simple,
    "logit_simple_female_or": or_female_simple,
    "logit_simple_error": logit_simple_error,
}

with open('analysis_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print(json.dumps(output, indent=2))
