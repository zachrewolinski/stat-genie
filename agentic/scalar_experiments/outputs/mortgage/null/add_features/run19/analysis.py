import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

# Load data
path = 'mortgage.csv'
df = pd.read_csv(path)

# Basic variables
# Use accept as outcome (1 accepted, 0 denied). If missing, fall back to deny.
if 'accept' in df.columns:
    outcome = 'accept'
elif 'deny' in df.columns:
    outcome = 'deny'
else:
    raise ValueError('No accept/deny column found')

# Use female indicator (1 female, 0 male)
if 'female' not in df.columns:
    raise ValueError('female column not found')

# Prepare clean dataset for unadjusted analysis
base_cols = ['female', outcome]
base = df[base_cols].dropna()

# Ensure binary
base = base[(base['female'].isin([0,1])) & (base[outcome].isin([0,1]))]

# Unadjusted acceptance/denial rates
rate_female = base.loc[base['female']==1, outcome].mean()
rate_male = base.loc[base['female']==0, outcome].mean()

# If outcome is deny, convert to accept for interpretability
if outcome == 'deny':
    rate_female = 1 - rate_female
    rate_male = 1 - rate_male

# Contingency table for chi-square
ct = pd.crosstab(base['female'], base[outcome])
chi2, p_chi2, dof, expected = stats.chi2_contingency(ct)

# Effect size (difference in acceptance rates)
rate_diff = rate_female - rate_male

# Logistic regression with controls
# Select control variables that are clearly related to creditworthiness and present in dataset
candidate_controls = [
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

controls = [c for c in candidate_controls if c in df.columns]
reg_cols = ['female', outcome] + controls
reg = df[reg_cols].dropna()

# Ensure binary outcome
reg = reg[(reg['female'].isin([0,1])) & (reg[outcome].isin([0,1]))]

# If outcome is deny, convert to accept for model (accept=1)
if outcome == 'deny':
    reg[outcome] = 1 - reg[outcome]

X = reg[['female'] + controls]
X = sm.add_constant(X, has_constant='add')
y = reg[outcome]

# Fit logistic regression
logit_model = sm.Logit(y, X)
result = logit_model.fit(disp=False)

# Extract female coefficient and p-value
female_coef = result.params.get('female', np.nan)
female_p = result.pvalues.get('female', np.nan)

# Convert to odds ratio
female_or = float(np.exp(female_coef)) if pd.notnull(female_coef) else np.nan
ci_low, ci_high = (np.nan, np.nan)
if pd.notnull(female_coef):
    ci = result.conf_int().loc['female']
    ci_low = float(np.exp(ci[0]))
    ci_high = float(np.exp(ci[1]))

# Sample sizes
n_total = len(base)
n_reg = len(reg)

print('N_total_unadjusted', n_total)
print('N_regression', n_reg)
print('Acceptance_rate_female', rate_female)
print('Acceptance_rate_male', rate_male)
print('Rate_diff_female_minus_male', rate_diff)
print('Chi2_p', p_chi2)
print('Female_logit_coef', female_coef)
print('Female_logit_OR', female_or)
print('Female_logit_OR_CI_low', ci_low)
print('Female_logit_OR_CI_high', ci_high)
print('Female_logit_p', female_p)
