import pandas as pd
import statsmodels.api as sm

# Load data
_df = pd.read_csv('mortgage.csv')

# Create a binary female indicator based on metadata (1 if female, 0 if male)
# The stored values appear to be noisy, so use a 0.5 threshold to map to 0/1.
_df['female_bin'] = (_df['female'] >= 0.5).astype(int)

# Simple approval rate comparison
rate_f = _df.loc[_df['female_bin'] == 1, 'accept'].mean()
rate_m = _df.loc[_df['female_bin'] == 0, 'accept'].mean()
rate_diff = rate_f - rate_m

# Logistic regression with controls
X = _df[[
    'female_bin', 'black', 'housing_expense_ratio', 'self_employed', 'married',
    'mortgage_credit', 'consumer_credit', 'bad_history', 'PI_ratio',
    'loan_to_value', 'denied_PMI'
]]
X = sm.add_constant(X)
y = _df['accept']

model = sm.Logit(y, X)
result = model.fit(disp=False)

# Also estimate with continuous female value (robustness)
X2 = _df[[
    'female', 'black', 'housing_expense_ratio', 'self_employed', 'married',
    'mortgage_credit', 'consumer_credit', 'bad_history', 'PI_ratio',
    'loan_to_value', 'denied_PMI'
]]
X2 = sm.add_constant(X2)
result2 = sm.Logit(y, X2).fit(disp=False)

# Print key results for inspection
print('N:', len(_df))
print('Female share (binary):', _df['female_bin'].mean())
print('Acceptance rate female:', rate_f)
print('Acceptance rate male:', rate_m)
print('Rate difference (female - male):', rate_diff)

print('\nLogit with female_bin coefficient:', result.params['female_bin'])
print('female_bin p-value:', result.pvalues['female_bin'])

print('\nLogit with continuous female coefficient:', result2.params['female'])
print('female (continuous) p-value:', result2.pvalues['female'])
