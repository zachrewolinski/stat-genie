import pandas as pd
import statsmodels.api as sm

# Load data
df = pd.read_csv('mortgage.csv')

# Basic cleanup
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# Outcome: accept (1=accepted, 0=denied)
# Ensure accept exists; fall back to 1 - deny if needed
if 'accept' not in df.columns and 'deny' in df.columns:
    df['accept'] = 1 - df['deny']

# Drop rows with missing key fields
key_cols = [
    'accept', 'female', 'black', 'housing_expense_ratio', 'self_employed', 'married',
    'mortgage_credit', 'consumer_credit', 'bad_history', 'PI_ratio', 'loan_to_value', 'denied_PMI'
]
key_cols = [c for c in key_cols if c in df.columns]

clean = df.dropna(subset=key_cols).copy()

# Logistic regression controlling for observed creditworthiness factors
X_cols = [
    'female', 'black', 'housing_expense_ratio', 'self_employed', 'married',
    'mortgage_credit', 'consumer_credit', 'bad_history', 'PI_ratio', 'loan_to_value', 'denied_PMI'
]
X_cols = [c for c in X_cols if c in clean.columns]
X = clean[X_cols]
X = sm.add_constant(X, has_constant='add')
y = clean['accept']

model = sm.Logit(y, X)
result = model.fit(disp=False)

# Extract female coefficient and p-value
female_coef = result.params.get('female', float('nan'))
female_p = result.pvalues.get('female', float('nan'))
female_ci = result.conf_int().loc['female'] if 'female' in result.params.index else None

print('Rows used:', len(clean))
print('\nLogit model results (female effect):')
print('female coef:', female_coef)
print('female p-value:', female_p)
if female_ci is not None:
    print('female 95% CI:', (female_ci[0], female_ci[1]))
