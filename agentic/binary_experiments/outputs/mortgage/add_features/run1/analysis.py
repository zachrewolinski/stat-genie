import pandas as pd
import numpy as np
import statsmodels.api as sm

df = pd.read_csv('mortgage.csv')

# Outcome: accept (1 accepted, 0 denied)
if 'accept' in df.columns:
    df['accept'] = df['accept']
else:
    df['accept'] = 1 - df['deny']

# Keep relevant columns
features = [
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

cols = ['accept'] + features

df_model = df[cols].copy()

# Drop rows with missing values in model variables
before = len(df_model)
df_model = df_model.dropna()
after = len(df_model)

# Descriptive: acceptance rates by gender
rate_by_gender = df_model.groupby('female')['accept'].mean()
count_by_gender = df_model.groupby('female')['accept'].count()

diff = rate_by_gender.loc[1.0] - rate_by_gender.loc[0.0] if 1.0 in rate_by_gender.index else np.nan

# Unadjusted logit
X_unadj = sm.add_constant(df_model[['female']])
model_unadj = sm.Logit(df_model['accept'], X_unadj).fit(disp=False)

# Adjusted logit
X_adj = sm.add_constant(df_model[features])
model_adj = sm.Logit(df_model['accept'], X_adj).fit(disp=False)

# Extract female effect
unadj_coef = model_unadj.params['female']
unadj_p = model_unadj.pvalues['female']

adj_coef = model_adj.params['female']
adj_p = model_adj.pvalues['female']

# Convert to odds ratios
unadj_or = float(np.exp(unadj_coef))
adj_or = float(np.exp(adj_coef))

print('Rows before dropna:', before)
print('Rows after dropna:', after)
print('Acceptance rate by gender (0=male,1=female):')
print(rate_by_gender)
print('Counts by gender:')
print(count_by_gender)
print('Acceptance rate difference (female - male):', diff)
print('\nUnadjusted logit (accept ~ female)')
print('coef:', unadj_coef, 'OR:', unadj_or, 'p:', unadj_p)
print('\nAdjusted logit (accept ~ female + controls)')
print('coef:', adj_coef, 'OR:', adj_or, 'p:', adj_p)
