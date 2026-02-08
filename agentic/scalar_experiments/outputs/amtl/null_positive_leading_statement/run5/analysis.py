import pandas as pd
import numpy as np
import patsy
import statsmodels.api as sm

# Load data
csv_path = 'amtl.csv'
df = pd.read_csv(csv_path)

# Prepare data
needed_cols = ['num_amtl', 'sockets', 'age', 'prob_male', 'tooth_class', 'genus']
df = df[needed_cols].copy()

# Drop rows with missing values
for col in needed_cols:
    df = df[df[col].notna()]

# Filter out rows with non-positive sockets or num_amtl > sockets
# (sockets represent opportunities for AMTL)
df = df[(df['sockets'] > 0) & (df['num_amtl'] >= 0) & (df['num_amtl'] <= df['sockets'])]

# Binary indicator for Homo sapiens vs non-human primates
human_label = 'Homo sapiens'
df['human'] = (df['genus'] == human_label).astype(int)

# Build design matrix
formula = 'human + age + prob_male + C(tooth_class)'
exog = patsy.dmatrix(formula, df, return_type='dataframe')

# Endogenous as successes and failures
success = df['num_amtl'].astype(float)
fail = (df['sockets'] - df['num_amtl']).astype(float)
endog = np.vstack([success, fail]).T

model = sm.GLM(endog, exog, family=sm.families.Binomial())
res = model.fit()

# Extract human effect
if 'human' in res.params.index:
    beta = float(res.params['human'])
    se = float(res.bse['human'])
    z = beta / se if se > 0 else 0.0
    p = float(res.pvalues['human'])
else:
    beta = 0.0
    se = np.nan
    z = 0.0
    p = 1.0

# Compute average marginal effect by predicting with human=1 vs human=0
# Use the same covariates for each row, change human
exog_h1 = exog.copy()
exog_h0 = exog.copy()
if 'human' in exog_h1.columns:
    exog_h1['human'] = 1
    exog_h0['human'] = 0

pred_h1 = res.predict(exog_h1)
pred_h0 = res.predict(exog_h0)

delta_p = float(np.mean(pred_h1 - pred_h0))

# Map evidence to Likert scale
abs_delta = abs(delta_p)

if p < 0.001 and abs_delta >= 0.05:
    base = 85
elif p < 0.01 and abs_delta >= 0.03:
    base = 75
elif p < 0.05 and abs_delta >= 0.02:
    base = 60
elif p < 0.1 and abs_delta >= 0.01:
    base = 45
elif abs_delta >= 0.005:
    base = 30
else:
    base = 10

score = base if delta_p > 0 else -base
score = int(max(-100, min(100, round(score))))

with open('conclusion.txt', 'w', encoding='utf-8') as f:
    f.write(str(score))

print({
    'beta_human': beta,
    'se_human': se,
    'z_human': z,
    'p_human': p,
    'delta_p': delta_p,
    'score': score,
    'n': len(df)
})
