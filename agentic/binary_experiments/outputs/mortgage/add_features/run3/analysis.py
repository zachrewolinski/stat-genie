import numpy as np
import pandas as pd
import statsmodels.api as sm

# Load data
_df = pd.read_csv('mortgage.csv')

# Core variables
outcome = 'accept'
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

# Keep relevant columns and drop missing
cols = [outcome, key_var] + controls
_df = _df[cols].dropna().copy()

# Descriptive approval rates by gender
rates = _df.groupby(key_var)[outcome].mean()
counts = _df[key_var].value_counts().sort_index()

# Unadjusted logistic regression: accept ~ female
X1 = sm.add_constant(_df[[key_var]])
model1 = sm.Logit(_df[outcome], X1).fit(disp=False)

# Adjusted logistic regression: accept ~ female + controls
X2 = sm.add_constant(_df[[key_var] + controls])
model2 = sm.Logit(_df[outcome], X2).fit(disp=False)

# Print key results
print('Approval rates by gender (0=male, 1=female):')
print(rates)
print('Counts by gender:')
print(counts)
print('\nUnadjusted Logit (accept ~ female):')
print(model1.summary())
print('\nAdjusted Logit (accept ~ female + controls):')
print(model2.summary())

# Save a small results table for quick reference
results = pd.DataFrame({
    'model': ['unadjusted', 'adjusted'],
    'female_coef': [model1.params[key_var], model2.params[key_var]],
    'female_pvalue': [model1.pvalues[key_var], model2.pvalues[key_var]],
    'female_odds_ratio': [model1.params[key_var].__float__(), model2.params[key_var].__float__()],
})
results['female_odds_ratio'] = results['female_coef'].apply(lambda x: float(np.exp(x)))
print('\nKey female effect (logit coef, p-value, odds ratio):')
print(results)
