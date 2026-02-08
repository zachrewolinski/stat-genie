import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = 'amtl.csv'
df = pd.read_csv(path)

# Basic cleaning
# Ensure sockets and num_amtl are non-negative and sockets >= num_amtl
# Drop rows with missing critical fields
required = ['num_amtl', 'sockets', 'age', 'prob_male', 'tooth_class', 'genus']
df = df.dropna(subset=required).copy()

# Filter any inconsistent rows (if any)
df = df[(df['sockets'] >= df['num_amtl']) & (df['sockets'] > 0)].copy()

# Create binary indicator for Homo sapiens
# 1 = Homo sapiens, 0 = non-human primates (Pan, Pongo, Papio)
df['human'] = (df['genus'] == 'Homo sapiens').astype(int)

# Build binomial response as successes/failures
# statsmodels GLM expects endog as 2-column array for binomial
# successes = num_amtl, failures = sockets - num_amtl

df['failures'] = df['sockets'] - df['num_amtl']

# Fit GLM with binomial family
# Include age, sex probability, and tooth class as covariates
formula = 'num_amtl + failures ~ human + age + prob_male + C(tooth_class)'
model = smf.glm(
    formula=formula,
    data=df,
    family=sm.families.Binomial()
)
result = model.fit()

# Extract coefficient for human effect
coef = result.params.get('human', np.nan)
se = result.bse.get('human', np.nan)

# Compute odds ratio and 95% CI
or_val = float(np.exp(coef)) if np.isfinite(coef) else np.nan
ci_low = float(np.exp(coef - 1.96 * se)) if np.isfinite(coef) and np.isfinite(se) else np.nan
ci_high = float(np.exp(coef + 1.96 * se)) if np.isfinite(coef) and np.isfinite(se) else np.nan

print('n_rows_used', len(df))
print('human_coef_logit', coef)
print('human_or', or_val)
print('human_or_ci', (ci_low, ci_high))
print('human_pvalue', float(result.pvalues.get('human', np.nan)))

# Also compare humans vs each genus by full genus model (optional)
formula2 = 'num_amtl + failures ~ C(genus) + age + prob_male + C(tooth_class)'
model2 = smf.glm(
    formula=formula2,
    data=df,
    family=sm.families.Binomial()
)
result2 = model2.fit()
print('\nFull genus model coefficients:')
print(result2.params)
print('\nFull genus model pvalues:')
print(result2.pvalues)
