import json
import pandas as pd
import statsmodels.api as sm
import numpy as np
from statsmodels.stats.proportion import proportions_ztest

# Load data

df = pd.read_csv('mortgage.csv')

# Variables
key_cols = ['accept', 'female']

covariates = [
    'black',
    'housing_expense_ratio',
    'self_employed',
    'married',
    'mortgage_credit',
    'consumer_credit',
    'bad_history',
    'PI_ratio',
    'loan_to_value',
    'denied_PMI',
]

available_covariates = [c for c in covariates if c in df.columns]

analysis_df = df[key_cols + available_covariates].dropna()

# Unadjusted approval rates
rates = analysis_df.groupby('female')['accept'].agg(['mean', 'count'])

male_rate = rates.loc[0, 'mean']
female_rate = rates.loc[1, 'mean']
male_n = rates.loc[0, 'count']
female_n = rates.loc[1, 'count']

count = np.array([female_rate * female_n, male_rate * male_n])
nobs = np.array([female_n, male_n])

zstat, pval = proportions_ztest(count, nobs)

# Logistic regression (adjusted)
X = analysis_df[['female'] + available_covariates]
X = sm.add_constant(X, has_constant='add')
y = analysis_df['accept']

logit_model = sm.Logit(y, X)
try:
    result = logit_model.fit(disp=False)
except Exception:
    result = logit_model.fit_regularized(disp=False)

if 'female' in result.params:
    coef = result.params['female']
    se = result.bse['female'] if hasattr(result, 'bse') and 'female' in result.bse else np.nan
    if np.isfinite(se):
        ci_low = coef - 1.96 * se
        ci_high = coef + 1.96 * se
        or_val = np.exp(coef)
        or_low = np.exp(ci_low)
        or_high = np.exp(ci_high)
    else:
        or_val = np.exp(coef)
        or_low = np.nan
        or_high = np.nan
    p_female = result.pvalues['female'] if hasattr(result, 'pvalues') and 'female' in result.pvalues else np.nan
else:
    coef = np.nan
    se = np.nan
    or_val = np.nan
    or_low = np.nan
    or_high = np.nan
    p_female = np.nan

summary = {
    'n_total': int(len(df)),
    'n_analysis': int(len(analysis_df)),
    'approval_rate_male': float(male_rate),
    'approval_rate_female': float(female_rate),
    'approval_rate_diff_female_minus_male': float(female_rate - male_rate),
    'proportion_test_z': float(zstat),
    'proportion_test_p': float(pval),
    'logit_female_coef': float(coef) if np.isfinite(coef) else None,
    'logit_female_se': float(se) if np.isfinite(se) else None,
    'logit_female_p': float(p_female) if np.isfinite(p_female) else None,
    'logit_female_or': float(or_val) if np.isfinite(or_val) else None,
    'logit_female_or_ci_low': float(or_low) if np.isfinite(or_low) else None,
    'logit_female_or_ci_high': float(or_high) if np.isfinite(or_high) else None,
    'covariates_used': ['female'] + available_covariates,
}

print(json.dumps(summary, indent=2))
