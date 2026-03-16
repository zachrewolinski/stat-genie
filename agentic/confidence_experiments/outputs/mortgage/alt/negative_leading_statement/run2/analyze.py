import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Load data
path = 'mortgage.csv'
df = pd.read_csv(path)

# Basic variables
# female: 1 if applicant is female, 0 if male
# deny: 1 if denied, 0 if accepted

# Remove rows with missing values in key columns
key_cols = [
    'female', 'deny', 'accept', 'black', 'housing_expense_ratio', 'self_employed',
    'married', 'mortgage_credit', 'consumer_credit', 'bad_history', 'PI_ratio',
    'loan_to_value', 'denied_PMI'
]

# Some datasets may not include both accept and deny, but here both exist.
use_cols = [c for c in key_cols if c in df.columns]

# Drop rows with missing data in the used columns
analysis_df = df[use_cols].dropna().copy()

# Ensure binary columns are numeric ints
for col in ['female', 'deny', 'accept', 'black', 'self_employed', 'married', 'bad_history', 'denied_PMI']:
    if col in analysis_df.columns:
        analysis_df[col] = analysis_df[col].astype(int)

# Unadjusted denial rates by gender
rates = analysis_df.groupby('female')['deny'].mean()
counts = analysis_df.groupby('female')['deny'].agg(['sum', 'count'])

# Two-proportion z-test for denial rate difference
# female=1 vs female=0
n_female = counts.loc[1, 'count']
rej_female = counts.loc[1, 'sum']
rate_female = rates.loc[1]

n_male = counts.loc[0, 'count']
rej_male = counts.loc[0, 'sum']
rate_male = rates.loc[0]

# pooled proportion
p_pool = (rej_female + rej_male) / (n_female + n_male)
se = np.sqrt(p_pool * (1 - p_pool) * (1/n_female + 1/n_male))
if se > 0:
    z = (rate_female - rate_male) / se
    p_z = 2 * (1 - stats.norm.cdf(abs(z)))
else:
    z = np.nan
    p_z = np.nan

# Logistic regression: deny ~ female + controls
# Controls: race, ratios, credit, etc.
controls = ['female']
for c in ['black', 'housing_expense_ratio', 'self_employed', 'married', 'mortgage_credit',
          'consumer_credit', 'bad_history', 'PI_ratio', 'loan_to_value', 'denied_PMI']:
    if c in analysis_df.columns:
        controls.append(c)

X = analysis_df[controls]
X = sm.add_constant(X, has_constant='add')
y = analysis_df['deny']

logit_model = sm.Logit(y, X)
try:
    logit_res = logit_model.fit(disp=False)
    logit_params = logit_res.params
    logit_pvalues = logit_res.pvalues
    logit_conf = logit_res.conf_int()
    # Odds ratio for female
    if 'female' in logit_params.index:
        or_female = float(np.exp(logit_params['female']))
        or_ci = np.exp(logit_conf.loc['female']).tolist()
        p_female = float(logit_pvalues['female'])
    else:
        or_female = np.nan
        or_ci = [np.nan, np.nan]
        p_female = np.nan
except Exception as e:
    logit_res = None
    or_female = np.nan
    or_ci = [np.nan, np.nan]
    p_female = np.nan

# Alternative model: accept ~ female + controls (should mirror deny)
if 'accept' in analysis_df.columns:
    y2 = analysis_df['accept']
    logit_model2 = sm.Logit(y2, X)
    try:
        logit_res2 = logit_model2.fit(disp=False)
        p_female_accept = float(logit_res2.pvalues.get('female', np.nan))
        or_female_accept = float(np.exp(logit_res2.params.get('female', np.nan)))
    except Exception:
        p_female_accept = np.nan
        or_female_accept = np.nan
else:
    p_female_accept = np.nan
    or_female_accept = np.nan

results = {
    'n_total': int(len(analysis_df)),
    'denial_rate_female': float(rate_female),
    'denial_rate_male': float(rate_male),
    'difference_denial_rate_female_minus_male': float(rate_female - rate_male),
    'z_test_p_value': float(p_z),
    'logit_female_odds_ratio_deny': float(or_female),
    'logit_female_or_ci_deny': [float(or_ci[0]), float(or_ci[1])],
    'logit_female_p_value_deny': float(p_female),
    'logit_female_odds_ratio_accept': float(or_female_accept),
    'logit_female_p_value_accept': float(p_female_accept),
}

print(json.dumps(results, indent=2))
