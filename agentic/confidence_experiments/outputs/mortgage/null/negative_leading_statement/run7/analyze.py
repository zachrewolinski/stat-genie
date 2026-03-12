import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = "mortgage.csv"
df = pd.read_csv(path)

# Basic checks
print(df.head())

# Binary outcome: accept (1 accepted, 0 denied)
# Ensure no missing
print(df[['accept','deny','female']].isna().sum())

# Bivariate: difference in acceptance rate by gender
rates = df.groupby('female')['accept'].mean()
print("accept rate by female", rates)

# Simple logit accept ~ female
model_simple = smf.logit("accept ~ female", data=df).fit(disp=False)
print(model_simple.summary())

# Multivariate logit controlling for covariates
covariates = [
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
formula = "accept ~ female + " + " + ".join(covariates)
model_full = smf.logit(formula, data=df).fit(disp=False)
print(model_full.summary())

# marginal effect: odds ratio for female
params = model_full.params
conf = model_full.conf_int()
or_female = np.exp(params['female'])
conf_or = np.exp(conf.loc['female'])
print("female OR", or_female, "CI", conf_or.tolist())

# Use robust standard errors? Not necessary but maybe check.
model_full_robust = model_full.get_robustcov_results(cov_type='HC1')
print(model_full_robust.summary())

# Evaluate predicted acceptance rates by gender controlling for covariates (average marginal effect)
try:
    ame = model_full.get_margeff(at='overall', method='dydx')
    print(ame.summary())
except Exception as e:
    print("margeff error", e)

