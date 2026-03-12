import json
import numpy as np
import pandas as pd
import statsmodels.api as sm

DATA_PATH = 'mortgage.csv'

# Load data
_df = pd.read_csv(DATA_PATH)

# Basic unadjusted denial rates by gender
rate_by_gender = _df.groupby('female')['deny'].mean().to_dict()
count_by_gender = _df['female'].value_counts().to_dict()

# Logistic regression: deny ~ female + controls
controls = [
    'black', 'housing_expense_ratio', 'self_employed', 'married',
    'mortgage_credit', 'consumer_credit', 'bad_history', 'PI_ratio',
    'loan_to_value', 'denied_PMI'
]

use_cols = ['deny', 'female'] + controls
model_df = _df[use_cols].dropna()

X = model_df[['female'] + controls]
X = sm.add_constant(X, has_constant='add')

y = model_df['deny']

logit = sm.Logit(y, X)
result = logit.fit(disp=False)

coef = result.params['female']
pval = result.pvalues['female']

or_female = np.exp(coef)
conf = result.conf_int().loc['female']
ci_lower, ci_upper = np.exp(conf[0]), np.exp(conf[1])

mean_X = X.mean()
mean_X_female0 = mean_X.copy()
mean_X_female1 = mean_X.copy()
mean_X_female0['female'] = 0
mean_X_female1['female'] = 1

prob0 = float(result.predict(mean_X_female0)[0])
prob1 = float(result.predict(mean_X_female1)[0])
prob_diff = prob1 - prob0

summary = {
    'rate_by_gender': rate_by_gender,
    'count_by_gender': count_by_gender,
    'coef_female': float(coef),
    'pval_female': float(pval),
    'odds_ratio_female': float(or_female),
    'ci_lower': float(ci_lower),
    'ci_upper': float(ci_upper),
    'prob0': prob0,
    'prob1': prob1,
    'prob_diff': float(prob_diff),
    'n_used': int(model_df.shape[0])
}

with open('analysis_results.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
