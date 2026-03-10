import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import chi2_contingency

# Load data
_df = pd.read_csv('mortgage.csv')

# Basic cleaning
# Ensure binary columns as numeric 0/1
binary_cols = ['female','black','self_employed','married','bad_history','deny','denied_PMI','accept']
for col in binary_cols:
    if col in _df.columns:
        _df[col] = pd.to_numeric(_df[col], errors='coerce')

# Drop rows with missing outcome or key predictor
_df_clean = _df.dropna(subset=['accept','female']).copy()

# Descriptive stats: approval rates by gender
rate_by_gender = _df_clean.groupby('female')['accept'].mean()
count_by_gender = _df_clean.groupby('female')['accept'].count()

# Contingency table for chi-square
ct = pd.crosstab(_df_clean['female'], _df_clean['accept'])
# Ensure columns 0/1 exist
ct = ct.reindex(index=[0.0,1.0], columns=[0.0,1.0], fill_value=0)
chi2, chi2_p, dof, expected = chi2_contingency(ct)

# Two-proportion z-test (female=1 vs female=0)
# successes = accept==1
successes = np.array([
    ct.loc[1.0,1.0],
    ct.loc[0.0,1.0]
])
# totals
nobs = np.array([
    ct.loc[1.0].sum(),
    ct.loc[0.0].sum()
])
# test difference in proportions
z_stat, z_p = proportions_ztest(successes, nobs)

# Logistic regression with controls
# Define predictors
predictors = [
    'female',
    'black',
    'housing_expense_ratio',
    'self_employed',
    'married',
    'mortgage_credit',
    'consumer_credit',
    'bad_history',
    'PI_ratio',
    'loan_to_value',
    'denied_PMI'
]

# Use only available predictors
predictors = [p for p in predictors if p in _df_clean.columns]

_df_model = _df_clean.dropna(subset=predictors + ['accept']).copy()

X = _df_model[predictors]
X = sm.add_constant(X, has_constant='add')
y = _df_model['accept']

def fit_logit(df, pred_list):
    pred_list = [p for p in pred_list if p in df.columns]
    df_model = df.dropna(subset=pred_list + ['accept']).copy()
    X_local = sm.add_constant(df_model[pred_list], has_constant='add')
    y_local = df_model['accept']
    model = sm.Logit(y_local, X_local)
    try:
        res = model.fit(disp=False)
        coef = res.params.get('female', np.nan)
        pval = res.pvalues.get('female', np.nan)
        odds = np.exp(coef) if pd.notna(coef) else np.nan
        nobs = int(res.nobs)
    except Exception:
        coef = np.nan
        pval = np.nan
        odds = np.nan
        nobs = int(len(df_model))
        res = None
    return {'coef': coef, 'pval': pval, 'odds': odds, 'nobs': nobs}

full_model = fit_logit(_df_model, predictors)

# Alternative model excluding denied_PMI (potential post-decision variable)
predictors_no_pmi = [p for p in predictors if p != 'denied_PMI']
no_pmi_model = fit_logit(_df_model, predictors_no_pmi)

# Save key results to json for inspection
results = {
    'n_total': int(len(_df_clean)),
    'n_male': int(count_by_gender.get(0.0, 0)),
    'n_female': int(count_by_gender.get(1.0, 0)),
    'approval_rate_male': float(rate_by_gender.get(0.0, np.nan)),
    'approval_rate_female': float(rate_by_gender.get(1.0, np.nan)),
    'chi2_p': float(chi2_p),
    'z_p': float(z_p),
    'logit_full_nobs': int(full_model['nobs']),
    'female_coef_full': float(full_model['coef']) if pd.notna(full_model['coef']) else None,
    'female_odds_ratio_full': float(full_model['odds']) if pd.notna(full_model['odds']) else None,
    'female_p_logit_full': float(full_model['pval']) if pd.notna(full_model['pval']) else None,
    'logit_no_pmi_nobs': int(no_pmi_model['nobs']),
    'female_coef_no_pmi': float(no_pmi_model['coef']) if pd.notna(no_pmi_model['coef']) else None,
    'female_odds_ratio_no_pmi': float(no_pmi_model['odds']) if pd.notna(no_pmi_model['odds']) else None,
    'female_p_logit_no_pmi': float(no_pmi_model['pval']) if pd.notna(no_pmi_model['pval']) else None,
}

print(json.dumps(results, indent=2))
