import pandas as pd
import numpy as np
import statsmodels.api as sm

# Load data
_df = pd.read_csv('mortgage.csv')

# Column mapping based on metadata descriptions
# female indicator
female_col = 'denied_PMI'  # description: 1 if applicant is female
# outcome: mortgage application denied
outcome_col = 'self_employed'  # description: 1 if application denied

# Covariates based on descriptions (creditworthiness and demographics)
# Using columns whose descriptions align with known underwriting factors
covariates = {
    'black_indicator': 'consumer_credit',        # description: 1 if applicant is Black
    'self_emp': 'accept',                        # description: 1 if self-employed
    'married': 'loan_to_value',                  # description: 1 if married
    'bad_credit': 'PI_ratio',                    # description: history of bad credit
    'mortgage_credit_score': 'married',          # description: mortgage credit score (1-4)
    'consumer_credit_score': 'black',            # description: consumer credit score (1-6)
    'housing_exp_ratio': 'mortgage_credit',      # description: housing expense / income
    'debt_income_ratio': 'housing_expense_ratio',# description: total debt / income
    'loan_to_value_ratio': 'Unnamed: 0',         # description: loan-to-value
    'denied_pmi': 'female',                      # description: denied PMI
}

cols = [outcome_col, female_col] + list(covariates.values())

# Drop rows with missing values in any used column
_df_model = _df[cols].dropna().copy()

# Build design matrix
X = pd.DataFrame({'female': _df_model[female_col].astype(float)})
for name, col in covariates.items():
    X[name] = _df_model[col].astype(float)

X = sm.add_constant(X)
y = _df_model[outcome_col].astype(float)

# Fit logistic regression
logit = sm.Logit(y, X)
result = logit.fit(disp=False)

# Extract female coefficient and p-value
coef = result.params['female']
pval = result.pvalues['female']

# Odds ratio
odds_ratio = np.exp(coef)

print('n_used', len(_df_model))
print('female_coef', coef)
print('female_odds_ratio', odds_ratio)
print('female_pval', pval)

# Also compute unadjusted difference in denial rate by gender
ct = pd.crosstab(_df_model[female_col], _df_model[outcome_col])
rates = ct.div(ct.sum(axis=1), axis=0)
rate_f = rates.loc[1,1] if 1 in rates.index else float('nan')
rate_m = rates.loc[0,1] if 0 in rates.index else float('nan')
print('denial_rate_female', rate_f)
print('denial_rate_male', rate_m)
print('diff_female_minus_male', rate_f - rate_m)
