import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

path = 'mortgage.csv'
df = pd.read_csv(path)

gender_col = 'denied_PMI'
denied_col = 'self_employed'

df['approved'] = 1 - df[denied_col]

models = {}

# unadjusted
models['unadjusted'] = 'approved ~ ' + gender_col

# adjusted all (except outcome/redundant)
all_covariates = [c for c in df.columns if c not in [denied_col, 'deny', 'approved']]
if gender_col not in all_covariates:
    all_covariates.append(gender_col)
models['all_covariates'] = 'approved ~ ' + ' + '.join(all_covariates)

# adjusted without 'accept'
cov_wo_accept = [c for c in all_covariates if c != 'accept']
models['no_accept'] = 'approved ~ ' + ' + '.join(cov_wo_accept)

# adjusted without 'accept' and without 'loan_to_value'
cov_wo_accept_ltv = [c for c in cov_wo_accept if c != 'loan_to_value']
models['no_accept_no_ltv'] = 'approved ~ ' + ' + '.join(cov_wo_accept_ltv)

# adjusted with selected continuous + key binaries
selected = [gender_col, 'bad_history', 'mortgage_credit', 'housing_expense_ratio', 'Unnamed: 0',
            'consumer_credit', 'PI_ratio', 'loan_to_value', 'married', 'black', 'accept', 'female']
selected = [c for c in selected if c in df.columns]
models['selected'] = 'approved ~ ' + ' + '.join(selected)

results = {}
for name, formula in models.items():
    try:
        model = smf.logit(formula, data=df).fit(disp=False, maxiter=200)
        params = model.params
        pvals = model.pvalues
        results[name] = {
            'coef': float(params.get(gender_col, np.nan)),
            'or': float(np.exp(params.get(gender_col, np.nan))),
            'p': float(pvals.get(gender_col, np.nan)),
            'converged': model.mle_retvals.get('converged', None)
        }
    except Exception as e:
        results[name] = {'error': str(e)}

for name, res in results.items():
    print(name, res)
