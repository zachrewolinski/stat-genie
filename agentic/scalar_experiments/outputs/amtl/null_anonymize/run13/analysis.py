import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('amtl.csv')

# Rename for clarity
_df = _df.rename(columns={
    'feature1': 'tooth_class',
    'feature2': 'specimen_id',
    'feature3': 'missing',
    'feature4': 'observable',
    'feature5': 'age',
    'feature6': 'age_unc',
    'feature7': 'sex',
    'feature8': 'genus',
    'feature9': 'region'
})

# Basic cleaning
_df = _df.copy()

# Ensure numeric
for col in ['missing','observable','age','sex']:
    _df[col] = pd.to_numeric(_df[col], errors='coerce')

# Keep valid rows
_df = _df.dropna(subset=['missing','observable','age','sex','tooth_class','genus'])

# Remove invalid counts
_df = _df[(_df['observable'] > 0) & (_df['missing'] >= 0) & (_df['missing'] <= _df['observable'])]

# Binary indicator for human
_df['is_human'] = (_df['genus'] == 'Homo sapiens').astype(int)

# Build successes / failures for binomial
_df['failures'] = _df['observable'] - _df['missing']

# Fit GLM binomial with counts
formula = 'missing + failures ~ is_human + age + sex + C(tooth_class)'

# statsmodels GLM accepts endog as 2-col array
endog = _df[['missing','failures']]
exog = sm.add_constant(pd.get_dummies(_df[['is_human','age','sex','tooth_class']], columns=['tooth_class'], drop_first=True))

model = sm.GLM(endog, exog, family=sm.families.Binomial())
res = model.fit()

# Extract coefficient and p-value for is_human
coef = res.params['is_human']
pval = res.pvalues['is_human']

# Convert log-odds to odds ratio
odds_ratio = float(np.exp(coef))

# Also compute predicted difference at mean covariates
mean_vals = exog.mean()
# Two scenarios
mean_vals_human = mean_vals.copy()
mean_vals_nonhuman = mean_vals.copy()
mean_vals_human['is_human'] = 1.0
mean_vals_nonhuman['is_human'] = 0.0

pred_human = res.predict(mean_vals_human)[0]
pred_nonhuman = res.predict(mean_vals_nonhuman)[0]

# Heuristic mapping to Likert [-100, 100]
# Use effect size and significance: log-odds magnitude and p-value
# Strong positive effect & significant -> large positive
# Strong negative effect & significant -> large negative
# weak or non-significant -> near 0

# Effect size scale: log-odds
abs_coef = abs(coef)

# Base score from effect size
# log-odds 0.0 -> 0, 0.2 -> 10, 0.5 -> 25, 1.0 -> 50, 1.5 -> 70, 2.0+ -> 85
if abs_coef < 0.05:
    base = 0
elif abs_coef < 0.2:
    base = 10
elif abs_coef < 0.5:
    base = 25
elif abs_coef < 1.0:
    base = 50
elif abs_coef < 1.5:
    base = 70
else:
    base = 85

# Adjust for p-value
if pval < 0.001:
    sig_factor = 1.0
elif pval < 0.01:
    sig_factor = 0.9
elif pval < 0.05:
    sig_factor = 0.75
elif pval < 0.1:
    sig_factor = 0.5
else:
    sig_factor = 0.25

score = int(round(base * sig_factor))

# Apply sign
if coef < 0:
    score = -score

# If effect direction is positive but very tiny and nonsignificant, keep near 0
if abs_coef < 0.05 and pval >= 0.1:
    score = 0

# Safety clamp
score = max(-100, min(100, score))

# Save conclusion
with open('conclusion.txt','w') as f:
    f.write(str(score))

# Also print some diagnostics for manual review if needed
print('coef_is_human', coef)
print('pval_is_human', pval)
print('odds_ratio', odds_ratio)
print('pred_human', pred_human)
print('pred_nonhuman', pred_nonhuman)
print('score', score)
