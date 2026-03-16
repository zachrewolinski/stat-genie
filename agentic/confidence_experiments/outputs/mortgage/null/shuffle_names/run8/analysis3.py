import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm


df = pd.read_csv('mortgage.csv')

# Identify key variables based on descriptions in info.json (shuffled names)
# female indicator is stored in column 'denied_PMI' per description
female_col = 'denied_PMI'
# acceptance indicator stored in column 'deny' per description
accept_col = 'deny'
# denial indicator stored in column 'self_employed' per description
# (should be roughly complement of accept)

print('female mean', df[female_col].mean(), 'accept mean', df[accept_col].mean())

# Drop rows with missing in these columns
sub = df[[female_col, accept_col]].dropna()

# crosstab
ct = pd.crosstab(sub[female_col], sub[accept_col])
print(ct)
chi2, p, dof, ex = stats.chi2_contingency(ct)
print('chi2', chi2, 'p', p)

# difference in acceptance rates
rates = sub.groupby(female_col)[accept_col].mean()
print('accept rates', rates.to_dict())

# logistic regression with controls (all other columns except target)
X_cols = [c for c in df.columns if c != accept_col]
# ensure female included
X = df[X_cols].copy()
X = sm.add_constant(X, has_constant='add')
# drop rows with missing
y = df[accept_col]
# keep rows with no missing in X or y
mask = X.notnull().all(axis=1) & y.notnull()
X2 = X[mask]
y2 = y[mask]

model = sm.Logit(y2, X2).fit(disp=False)
print(model.summary2().tables[1].loc[[female_col]])
