import pandas as pd
import numpy as np
import statsmodels.api as sm

_df = pd.read_csv('mortgage.csv')

# Use metadata mapping
gender_col = 'denied_PMI'  # female indicator
outcome_denied = 'self_employed'  # denied indicator

# choose subset of controls with many unique values
candidate_controls = ['bad_history','mortgage_credit','housing_expense_ratio','Unnamed: 0','married','black']
controls = [c for c in candidate_controls if c in _df.columns]

cols = [gender_col] + controls
X = _df[cols].copy()
y = _df[outcome_denied]

mask = X.notna().all(axis=1) & y.notna()
X = X[mask]
y = y[mask]

X = sm.add_constant(X, has_constant='add')

try:
    model = sm.Logit(y, X)
    res = model.fit(disp=False)
    print(res.summary())
    print('female coef', res.params[gender_col], 'p', res.pvalues[gender_col], 'OR', float(np.exp(res.params[gender_col])))
except Exception as e:
    print('logit error', e)
