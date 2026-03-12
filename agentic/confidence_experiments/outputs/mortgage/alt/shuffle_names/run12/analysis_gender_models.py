import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Load data
path = 'mortgage.csv'
df = pd.read_csv(path)

female = df['denied_PMI']  # female indicator

deny = df['self_employed']  # deny indicator
approve = 1 - deny

analysis_df = pd.DataFrame({
    'female': female,
    'deny': deny,
    'approve': approve,
    'consumer_credit': df['consumer_credit'],
    'mortgage_credit': df['mortgage_credit'],
    'accept': df['accept'],
    'loan_to_value': df['loan_to_value'],
    'married': df['married'],
    'black': df['black'],
    'PI_ratio': df['PI_ratio'],
    'housing_expense_ratio': df['housing_expense_ratio'],
    'loan_to_value_ratio': df['Unnamed: 0'],
    'denied_PMI': df['female'],
})

analysis_df = analysis_df.replace([np.inf, -np.inf], np.nan).dropna()

female = analysis_df['female']
deny = analysis_df['deny']
approve = analysis_df['approve']

# Basic rates
rate_female = approve[female == 1].mean()
rate_male = approve[female == 0].mean()
rate_gap = rate_female - rate_male

# Chi-square test
cont = pd.crosstab(female, approve)
chi2, p_chi, dof, expected = stats.chi2_contingency(cont)

print('N used:', len(analysis_df))
print('Approval rate female:', rate_female)
print('Approval rate male:', rate_male)
print('Rate gap (female - male):', rate_gap)
print('Chi-square p-value:', p_chi)

# Helper to fit logit and print female effect

def fit_model(cols, label):
    X = analysis_df[cols].copy()
    X = sm.add_constant(pd.concat([female.rename('female'), X], axis=1))
    model = sm.Logit(deny, X).fit(disp=False)
    coef = model.params['female']
    se = model.bse['female']
    p = model.pvalues['female']
    or_val = np.exp(coef)
    print(f"\n{label}")
    print('coef', coef, 'SE', se, 'OR', or_val, 'p', p)

# Model A: unadjusted
model_unadj = sm.Logit(deny, sm.add_constant(female)).fit(disp=False)
coef = model_unadj.params['female']
se = model_unadj.bse['female']
p = model_unadj.pvalues['female']
or_val = np.exp(coef)
print('\nUnadjusted')
print('coef', coef, 'SE', se, 'OR', or_val, 'p', p)

# Model B: controls (credit + demographics) excluding denied_PMI indicator
cols_b = [
    'consumer_credit',
    'mortgage_credit',
    'accept',
    'loan_to_value',
    'married',
    'black',
    'PI_ratio',
    'housing_expense_ratio',
    'loan_to_value_ratio'
]
fit_model(cols_b, 'Adjusted (no denied_PMI control)')

# Model C: credit ratios & scores only
cols_c = [
    'mortgage_credit',
    'housing_expense_ratio',
    'loan_to_value_ratio',
    'married',
    'black',
    'PI_ratio'
]
fit_model(cols_c, 'Adjusted (credit ratios/scores only)')

# Model D: add race and marital, exclude self-employed
cols_d = [
    'consumer_credit',
    'mortgage_credit',
    'loan_to_value',
    'married',
    'black',
    'PI_ratio',
    'housing_expense_ratio',
    'loan_to_value_ratio'
]
fit_model(cols_d, 'Adjusted (exclude self-employed)')
