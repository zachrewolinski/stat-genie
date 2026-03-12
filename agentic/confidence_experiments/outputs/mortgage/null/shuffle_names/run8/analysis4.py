import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

df = pd.read_csv('mortgage.csv')

female_col = 'denied_PMI'
denial_col = 'self_employed'  # per description

sub = df[[female_col, denial_col]].dropna()
ct = pd.crosstab(sub[female_col], sub[denial_col])
print(ct)
chi2, p, dof, ex = stats.chi2_contingency(ct)
print('chi2', chi2, 'p', p)

rates = sub.groupby(female_col)[denial_col].mean()
print('denial rates', rates.to_dict())

# logit with controls on denial outcome
X_cols = [c for c in df.columns if c != denial_col]
X = df[X_cols].copy()
X = sm.add_constant(X, has_constant='add')
mask = X.notnull().all(axis=1) & df[denial_col].notnull()
X2 = X[mask]
y2 = df[denial_col][mask]
model = sm.Logit(y2, X2).fit(disp=False)
print(model.summary2().tables[1].loc[[female_col]])
