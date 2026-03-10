import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# Load data
path = 'mortgage.csv'
df = pd.read_csv(path)

# Basic cleaning: ensure binary columns are numeric 0/1
for col in ['female', 'deny', 'accept', 'black', 'self_employed', 'married', 'bad_history', 'denied_PMI']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Use deny as outcome (1=denied, 0=accepted)
# Drop rows with missing relevant values
cols_needed = ['female', 'deny']
df_simple = df[cols_needed].dropna()

# Contingency table and chi-square
ct = pd.crosstab(df_simple['female'], df_simple['deny'])
# Ensure order: female 0/1, deny 0/1
ct = ct.reindex(index=[0,1], columns=[0,1])
chi2, p_chi, dof, expected = stats.chi2_contingency(ct)

# Approval/denial rates by gender
rates = df_simple.groupby('female')['deny'].agg(['mean','count'])

# Logistic regression without controls
model_simple = smf.logit('deny ~ female', data=df_simple).fit(disp=False)

# Logistic regression with controls
control_vars = [
    'black', 'housing_expense_ratio', 'self_employed', 'married',
    'mortgage_credit', 'consumer_credit', 'bad_history', 'PI_ratio',
    'loan_to_value', 'denied_PMI'
]
cols_full = ['deny', 'female'] + [c for c in control_vars if c in df.columns]

# Drop missing rows for full model
full_df = df[cols_full].dropna()

formula = 'deny ~ female'
if len(cols_full) > 2:
    formula += ' + ' + ' + '.join([c for c in cols_full if c not in ['deny', 'female']])

model_full = smf.logit(formula, data=full_df).fit(disp=False)

# Compute odds ratio for female
coef_simple = model_simple.params['female']
se_simple = model_simple.bse['female']

coef_full = model_full.params['female']
se_full = model_full.bse['female']

# OR and 95% CI
or_simple = np.exp(coef_simple)
ci_simple = np.exp(coef_simple + np.array([-1,1]) * 1.96 * se_simple)

or_full = np.exp(coef_full)
ci_full = np.exp(coef_full + np.array([-1,1]) * 1.96 * se_full)

# Marginal effect (difference in predicted denial probability) at mean of covariates
# For simple model, use logistic of intercept and intercept+female
intercept = model_simple.params['Intercept']
prob_male = 1/(1+np.exp(-(intercept)))
prob_female = 1/(1+np.exp(-(intercept + coef_simple)))

# For full model: compute average predicted probability by toggling female
full_df_mean = full_df.copy()
# Predict using model_full
pred_male = model_full.predict(full_df_mean.assign(female=0))
pred_female = model_full.predict(full_df_mean.assign(female=1))
mean_diff_full = (pred_female - pred_male).mean()

print('N total:', len(df))
print('N simple:', len(df_simple))
print('N full:', len(full_df))
print('Contingency table (female x deny):')
print(ct)
print('Chi-square:', chi2, 'p=', p_chi)
print('Denial rates by female:')
print(rates)
print('\nSimple logit (deny ~ female):')
print(model_simple.summary())
print('OR simple:', or_simple, 'CI:', ci_simple)
print('Predicted deny prob male:', prob_male, 'female:', prob_female, 'diff:', prob_female - prob_male)

print('\nFull logit (with controls):')
print(model_full.summary())
print('OR full:', or_full, 'CI:', ci_full)
print('Average marginal effect (female=1 vs 0):', mean_diff_full)
