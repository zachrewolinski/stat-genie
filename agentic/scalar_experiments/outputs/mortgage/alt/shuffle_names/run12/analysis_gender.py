import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Load data
path = 'mortgage.csv'
df = pd.read_csv(path)

# Map variables based on info.json descriptions
female = df['denied_PMI']  # description says 1 if applicant is female

deny = df['self_employed']  # description says 1 if mortgage application was denied
approve = 1 - deny

# Assemble dataframe for analysis, drop missing
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

# Chi-square test of independence
cont = pd.crosstab(female, approve)
chi2, p_chi, dof, expected = stats.chi2_contingency(cont)

# Logistic regression: deny ~ female (unadjusted)
logit_unadj = sm.Logit(deny, sm.add_constant(female)).fit(disp=False)

# Adjusted model with available controls
control_cols = [
    'consumer_credit',      # black indicator
    'mortgage_credit',      # housing expense ratio
    'accept',               # self-employed indicator (per description)
    'loan_to_value',        # married indicator
    'married',              # mortgage credit score
    'black',                # consumer credit score
    'PI_ratio',             # bad credit history
    'housing_expense_ratio',# debt-to-income ratio
    'loan_to_value_ratio',  # loan-to-value ratio
    'denied_PMI'            # denied PMI indicator
]

X = analysis_df[control_cols].copy()
X = sm.add_constant(pd.concat([female.rename('female'), X], axis=1))

logit_adj = sm.Logit(deny, X).fit(disp=False)

# Extract coefficient and p-value for female
coef_unadj = logit_unadj.params['female']
se_unadj = logit_unadj.bse['female']
p_unadj = logit_unadj.pvalues['female']

coef_adj = logit_adj.params['female']
se_adj = logit_adj.bse['female']
p_adj = logit_adj.pvalues['female']

# Odds ratios
or_unadj = np.exp(coef_unadj)
or_adj = np.exp(coef_adj)

# Output summary
print('N used:', len(analysis_df))
print('Counts by gender (female=1):')
print(cont)
print('\nApproval rate female:', rate_female)
print('Approval rate male:', rate_male)
print('Rate gap (female - male):', rate_gap)
print('\nChi-square p-value:', p_chi)
print('\nLogit unadjusted coef (female on deny):', coef_unadj, 'SE', se_unadj, 'OR', or_unadj, 'p', p_unadj)
print('Logit adjusted coef (female on deny):', coef_adj, 'SE', se_adj, 'OR', or_adj, 'p', p_adj)
