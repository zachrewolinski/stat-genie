import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import chi2_contingency

# Load data
path = 'mortgage.csv'
df = pd.read_csv(path)

female = 'feature2'
accepted = 'feature14'

# Drop rows with missing female or accepted
base_df = df[[female, accepted]].dropna()

print('rows', len(df))
print('rows after dropna', len(base_df))

# Acceptance rate by gender
rate = base_df.groupby(female)[accepted].mean()
count = base_df.groupby(female)[accepted].agg(['count','sum'])
print('acceptance rate by female', rate.to_dict())
print('counts', count)

# contingency table and chi-square
ct = pd.crosstab(base_df[female], base_df[accepted])
chi2, p, dof, exp = chi2_contingency(ct)
print('chi2', chi2, 'p', p)

# Logistic regression unadjusted
model = sm.Logit(base_df[accepted], sm.add_constant(base_df[[female]])).fit(disp=0)
print(model.summary())

# Adjusted model with other covariates
cols = [c for c in df.columns if c not in [accepted, 'feature11']]
# drop rows with missing in any of these
adj_df = df[cols + [accepted]].dropna()
X = sm.add_constant(adj_df[cols])
model2 = sm.Logit(adj_df[accepted], X).fit(disp=0, maxiter=200)
print(model2.summary())

coef = model2.params[female]
pval = model2.pvalues[female]
print('adjusted female coef', coef, 'p', pval)

# Odds ratios
or_unadj = np.exp(model.params[female])
or_adj = np.exp(coef)
print('OR unadj', or_unadj)
print('OR adj', or_adj)

