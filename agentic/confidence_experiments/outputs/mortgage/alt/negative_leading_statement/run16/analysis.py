import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
df = pd.read_csv('mortgage.csv')

# Drop unnamed index if present
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# Basic cleaning: drop rows with missing values in analysis columns
analysis_cols = ['deny', 'accept', 'female', 'black', 'housing_expense_ratio', 'self_employed', 'married',
                 'mortgage_credit', 'consumer_credit', 'bad_history', 'PI_ratio', 'loan_to_value', 'denied_PMI']
analysis_cols = [c for c in analysis_cols if c in df.columns]

clean = df.dropna(subset=analysis_cols)

# Check accept/deny consistency if both exist
consistency = None
if 'accept' in clean.columns and 'deny' in clean.columns:
    consistency = float((clean['accept'] + clean['deny'] == 1).mean())

# Descriptive denial rates by gender
rate_by_gender = (
    clean.groupby('female')['deny']
    .agg(['mean', 'count'])
    .rename(index={0:'male', 1:'female'})
)

# Unadjusted logit
unadj_model = smf.logit('deny ~ female', data=clean).fit(disp=False)

# Adjusted logit with covariates (exclude accept to avoid leakage)
control_vars = ['female']
for c in ['black','housing_expense_ratio','self_employed','married','mortgage_credit','consumer_credit',
          'bad_history','PI_ratio','loan_to_value','denied_PMI']:
    if c in clean.columns:
        control_vars.append(c)

formula = 'deny ~ ' + ' + '.join(control_vars)
adj_model = smf.logit(formula, data=clean).fit(disp=False)

# Extract female effect
unadj_coef = unadj_model.params['female']
unadj_p = unadj_model.pvalues['female']
unadj_or = float(np.exp(unadj_coef))

adj_coef = adj_model.params['female']
adj_p = adj_model.pvalues['female']
adj_or = float(np.exp(adj_coef))

# Effect sizes (difference in denial rates)
rate_male = rate_by_gender.loc['male','mean']
rate_female = rate_by_gender.loc['female','mean']
rate_diff = float(rate_female - rate_male)

results = {
    'n': int(clean.shape[0]),
    'deny_rate_male': float(rate_male),
    'deny_rate_female': float(rate_female),
    'deny_rate_diff_female_minus_male': rate_diff,
    'unadjusted': {
        'coef_female': float(unadj_coef),
        'p_value': float(unadj_p),
        'odds_ratio': unadj_or,
    },
    'adjusted': {
        'coef_female': float(adj_coef),
        'p_value': float(adj_p),
        'odds_ratio': adj_or,
    },
    'accept_deny_consistency': consistency,
}

print(json.dumps(results, indent=2))
