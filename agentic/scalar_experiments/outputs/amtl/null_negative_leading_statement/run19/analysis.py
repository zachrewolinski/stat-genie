import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from patsy import dmatrix

# Load data
info = json.load(open('info.json'))
df = pd.read_csv('amtl.csv')

# Basic cleaning
# Keep rows with valid counts and covariates
needed = ['num_amtl', 'sockets', 'age', 'prob_male', 'tooth_class', 'genus']
df = df.dropna(subset=needed).copy()
# Ensure counts are valid
mask = (df['sockets'] > 0) & (df['num_amtl'] >= 0) & (df['num_amtl'] <= df['sockets'])
df = df.loc[mask].copy()

# Human indicator
# Normalize genus values
# Expect 'Homo sapiens' for humans
human_label = 'Homo sapiens'
df['human'] = (df['genus'] == human_label).astype(int)

# Model: binomial GLM with counts
# Use two-column endog: successes and failures
endog = np.column_stack([df['num_amtl'].values, (df['sockets'] - df['num_amtl']).values])

# Build design matrix via formula
# Keep tooth_class categorical
formula = 'human + age + prob_male + C(tooth_class)'
exog = dmatrix(formula, df, return_type='dataframe')
model = sm.GLM(endog, exog, family=sm.families.Binomial())
res = model.fit()

# Extract human effect
beta = res.params.get('human', np.nan)
se = res.bse.get('human', np.nan)
pval = res.pvalues.get('human', np.nan)

# Average marginal difference in predicted probability
# Compute predictions with human set to 1 and 0, holding others fixed
exog = res.model.exog.copy()
exog_h = exog.copy()
exog_n = exog.copy()
# Identify column index for 'human'
col_names = res.model.exog_names
if 'human' in col_names:
    idx = col_names.index('human')
    exog_h[:, idx] = 1
    exog_n[:, idx] = 0

pred_h = res.predict(exog_h)
pred_n = res.predict(exog_n)
mean_diff = float(np.mean(pred_h - pred_n))

# Map to Likert scale
# Use odds ratio and p-value to determine strength
if np.isnan(beta) or np.isnan(pval):
    score = 0
else:
    oratio = float(np.exp(beta))
    # Direction
    direction = 1 if beta > 0 else -1 if beta < 0 else 0

    # Strength tiers based on OR and p-value
    if pval < 0.01:
        if oratio >= 2.0:
            base = 90
        elif oratio >= 1.5 or oratio <= (1/1.5):
            base = 80
        elif oratio >= 1.2 or oratio <= (1/1.2):
            base = 65
        else:
            base = 50
    elif pval < 0.05:
        if oratio >= 1.5 or oratio <= (1/1.5):
            base = 70
        elif oratio >= 1.2 or oratio <= (1/1.2):
            base = 55
        else:
            base = 40
    elif pval < 0.2:
        base = 25
    else:
        base = 10

    # If effect is tiny in absolute probability, dampen
    if abs(mean_diff) < 0.01:
        base = min(base, 20)
    elif abs(mean_diff) < 0.03:
        base = min(base, 40)

    score = int(round(direction * base))

# Clamp to [-100, 100]
score = int(max(-100, min(100, score)))

# Save conclusion
with open('conclusion.txt', 'w') as f:
    f.write(str(score))

# Print summary for sanity
print('Rows:', len(df))
print('Human coef:', beta)
print('Human pval:', pval)
print('Human OR:', np.exp(beta) if not np.isnan(beta) else np.nan)
print('Mean diff (human - nonhuman):', mean_diff)
print('Score:', score)
