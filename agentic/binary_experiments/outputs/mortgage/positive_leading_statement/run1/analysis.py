import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
_df = pd.read_csv('mortgage.csv')

# Variables
outcome = 'deny'
key_var = 'female'
controls = [
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

# Basic rate comparison
basic_df = _df[[outcome, key_var]].dropna()
rate_by_gender = basic_df.groupby(key_var)[outcome].mean()
print('Denial rate by gender (0=male,1=female):')
print(rate_by_gender)

# Unadjusted logit
X_basic = sm.add_constant(basic_df[[key_var]])
model_basic = sm.Logit(basic_df[outcome], X_basic).fit(disp=0)
print('\nUnadjusted logit: deny ~ female')
print(model_basic.summary())
print('Unadjusted OR (female):', float(np.exp(model_basic.params[key_var])))
print('Unadjusted p-value (female):', float(model_basic.pvalues[key_var]))

# Adjusted logit with controls
adj_df = _df[[outcome, key_var] + controls].dropna()
X_adj = sm.add_constant(adj_df[[key_var] + controls])
model_adj = sm.Logit(adj_df[outcome], X_adj).fit(disp=0)
print('\nAdjusted logit: deny ~ female + controls')
print(model_adj.summary())
print('Adjusted OR (female):', float(np.exp(model_adj.params[key_var])))
print('Adjusted p-value (female):', float(model_adj.pvalues[key_var]))

# Save key results for quick inspection
results = {
    'n_basic': int(basic_df.shape[0]),
    'n_adjusted': int(adj_df.shape[0]),
    'deny_rate_male': float(rate_by_gender.loc[0.0]),
    'deny_rate_female': float(rate_by_gender.loc[1.0]),
    'unadjusted_or_female': float(np.exp(model_basic.params[key_var])),
    'unadjusted_p_female': float(model_basic.pvalues[key_var]),
    'adjusted_or_female': float(np.exp(model_adj.params[key_var])),
    'adjusted_p_female': float(model_adj.pvalues[key_var]),
}
print('\nKey results:', results)
