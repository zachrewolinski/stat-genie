import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest

path = 'mortgage.csv'

df = pd.read_csv(path)

# According to info.json descriptions (names are shuffled),
# 'denied_PMI' corresponds to gender: 1 if female, 0 if male.
# 'deny' corresponds to approval: 1 if accepted, 0 if denied.

# Select gender and approval columns
gender_col = 'denied_PMI'
approval_col = 'deny'

# Drop missing in relevant columns
sub = df[[gender_col, approval_col]].dropna()

# Convert to int for clarity
sub[gender_col] = sub[gender_col].astype(int)
sub[approval_col] = sub[approval_col].astype(int)

# Approval rates by gender
approval_rates = sub.groupby(gender_col)[approval_col].mean()
counts = sub.groupby(gender_col)[approval_col].agg(['sum','count'])

# Two-proportion z test (approval rates)
count = np.array([counts.loc[1, 'sum'] if 1 in counts.index else 0,
                  counts.loc[0, 'sum'] if 0 in counts.index else 0])
obs = np.array([counts.loc[1, 'count'] if 1 in counts.index else 0,
                counts.loc[0, 'count'] if 0 in counts.index else 0])

# Guard against missing category
zstat = pval = None
if obs.min() > 0:
    zstat, pval = proportions_ztest(count, obs)

print('Approval rates by gender (1=female,0=male):')
print(approval_rates)
print('Counts (approved, total) by gender:')
print(counts)
print('Two-proportion z-test (female vs male approval):', zstat, pval)

# Logistic regression: approval ~ gender (unadjusted)
X = sm.add_constant(sub[[gender_col]])
model = sm.Logit(sub[approval_col], X)
res = model.fit(disp=False)
print('\nLogit approval ~ gender')
print(res.summary())
print('Odds ratio for female:', np.exp(res.params[gender_col]))

# Multivariate logistic regression controlling for other variables
# Exclude outcome and gender and obvious ID-like column (bad_history)
exclude_cols = {approval_col, gender_col, 'bad_history'}
# Use remaining columns
features = [c for c in df.columns if c not in exclude_cols]

# Prepare data (drop rows with missing in any selected columns)
reg_df = df[[approval_col, gender_col] + features].dropna()

X_multi = reg_df[[gender_col] + features]
# Add constant
X_multi = sm.add_constant(X_multi, has_constant='add')

y_multi = reg_df[approval_col].astype(int)

model_multi = sm.Logit(y_multi, X_multi)
res_multi = model_multi.fit(disp=False)
print('\nLogit approval ~ gender + controls')
print(res_multi.summary())
print('Odds ratio for female (adjusted):', np.exp(res_multi.params[gender_col]))
