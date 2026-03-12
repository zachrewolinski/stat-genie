import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Load data
path = 'mortgage.csv'
df = pd.read_csv(path)

# Identify columns based on inference from distributions
# Outcome: denial indicator appears to be 'self_employed' (mean ~0.12)
# Acceptance indicator appears to be 'deny' (mean ~0.88)
# Gender indicator appears to be 'denied_PMI' (mean ~0.21)

outcome = 'self_employed'  # actual denial
accept_col = 'deny'        # actual acceptance
female_col = 'denied_PMI'  # inferred female

# Basic counts and rates
rate_by_gender = df.groupby(female_col)[outcome].agg(['mean','count'])
print('Denial rate by gender (1=female):')
print(rate_by_gender)

# Difference in denial rates and two-proportion z-test
p1 = rate_by_gender.loc[1, 'mean'] if 1 in rate_by_gender.index else np.nan
p0 = rate_by_gender.loc[0, 'mean'] if 0 in rate_by_gender.index else np.nan
n1 = rate_by_gender.loc[1, 'count'] if 1 in rate_by_gender.index else np.nan
n0 = rate_by_gender.loc[0, 'count'] if 0 in rate_by_gender.index else np.nan

if not np.isnan([p1,p0,n1,n0]).any():
    # pooled proportion
    p_pool = (p1*n1 + p0*n0) / (n1 + n0)
    se = np.sqrt(p_pool*(1-p_pool)*(1/n1 + 1/n0))
    z = (p1 - p0) / se if se > 0 else np.nan
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    print(f"\nDifference in denial rates (female - male): {p1 - p0:.4f}")
    print(f"Two-proportion z-test: z={z:.3f}, p={p_value:.4g}")

# Logistic regression: denial on gender + controls
# Exclude ID-like column and acceptance complement to avoid perfect collinearity
controls = [
    'consumer_credit',
    'mortgage_credit',
    'accept',
    'loan_to_value',
    'married',
    'black',
    'PI_ratio',
    'housing_expense_ratio',
    'Unnamed: 0',
    'female'  # this is likely denied PMI
]

cols_for_model = [female_col] + controls + [outcome]
df_model = df[cols_for_model].replace([np.inf, -np.inf], np.nan).dropna()
X = df_model[[female_col] + controls].copy()
X = sm.add_constant(X, has_constant='add')
y = df_model[outcome]

model = sm.Logit(y, X)
res = model.fit(disp=False)
print('\nLogit results (denial outcome):')
print(res.summary())

# Extract gender coefficient
coef = res.params[female_col]
se = res.bse[female_col]
pval = res.pvalues[female_col]

odds_ratio = np.exp(coef)
print(f"\nGender coefficient (female): {coef:.4f}, SE={se:.4f}, p={pval:.4g}, OR={odds_ratio:.3f}")

# Average marginal effect on denial probability
X_f1 = X.copy()
X_f0 = X.copy()
X_f1[female_col] = 1
X_f0[female_col] = 0
pred1 = res.predict(X_f1)
pred0 = res.predict(X_f0)
ame = (pred1 - pred0).mean()
print(f"Average marginal effect on denial probability (female vs male): {ame:.4f}")

# Also compute effect on approval (acceptance) rates directly
accept_rate_by_gender = df.groupby(female_col)[accept_col].agg(['mean','count'])
print('\nAcceptance rate by gender (1=female):')
print(accept_rate_by_gender)
