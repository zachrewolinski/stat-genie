import pandas as pd
import numpy as np
import statsmodels.api as sm

_df = pd.read_csv('mortgage.csv')
if 'Unnamed: 0' in _df.columns:
    df = _df.drop(columns=['Unnamed: 0'])
else:
    df = _df.copy()

control_cols = ['female','black','housing_expense_ratio','self_employed','married',
                'mortgage_credit','consumer_credit','bad_history','PI_ratio',
                'loan_to_value','denied_PMI']

adj_df = df[control_cols + ['deny']].dropna()
X = sm.add_constant(adj_df[control_cols])
model = sm.GLM(adj_df['deny'], X, family=sm.families.Binomial()).fit()
coef = model.params['female']
or_val = np.exp(coef)
pval = model.pvalues['female']
print({'coef':float(coef),'or':float(or_val),'p':float(pval)})
