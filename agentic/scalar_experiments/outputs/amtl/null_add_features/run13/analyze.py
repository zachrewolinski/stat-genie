import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('amtl.csv')

# Keep relevant columns
cols = ['num_amtl','sockets','age','prob_male','tooth_class','genus']
missing = [c for c in cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

sub = df[cols].copy()

# Drop rows with missing or invalid values
sub = sub.dropna()
# Ensure sockets > 0
sub = sub[sub['sockets'] > 0]

# Standardize categorical values
sub['genus'] = sub['genus'].astype(str)
sub['tooth_class'] = sub['tooth_class'].astype(str)

# Create indicator for Homo sapiens
sub['is_human'] = (sub['genus'].str.strip() == 'Homo sapiens').astype(int)

# Keep only rows with genus in expected set to avoid garbage
expected = {'Homo sapiens','Pan','Pongo','Papio'}
sub = sub[sub['genus'].str.strip().isin(expected)].copy()

# Outcome as proportion with binomial weights
sub['amtl_rate'] = sub['num_amtl'] / sub['sockets']

# Fit GLM: amtl_rate ~ is_human + age + prob_male + tooth_class
# Using C(tooth_class) for categorical
model = smf.glm(
    formula='amtl_rate ~ is_human + age + prob_male + C(tooth_class)',
    data=sub,
    family=sm.families.Binomial(),
    var_weights=sub['sockets']
).fit()

# Extract coefficient for is_human
coef = model.params.get('is_human', np.nan)
se = model.bse.get('is_human', np.nan)

# Compute odds ratio and p-value
or_val = float(np.exp(coef)) if np.isfinite(coef) else np.nan
pval = model.pvalues.get('is_human', np.nan)

# Simple scalar mapping based on sign and p-value
# Start with base magnitude from effect size (log-odds)
# clamp log-odds to +/-2 for scaling
if np.isfinite(coef):
    effect_mag = min(abs(coef), 2.0) / 2.0  # 0-1
else:
    effect_mag = 0.0

if np.isfinite(pval):
    if pval < 0.001:
        sig = 1.0
    elif pval < 0.01:
        sig = 0.85
    elif pval < 0.05:
        sig = 0.7
    elif pval < 0.1:
        sig = 0.55
    else:
        sig = 0.4
else:
    sig = 0.4

raw = effect_mag * sig
# Convert to 0-100 and apply sign
scalar = int(round(100 * raw * (1 if coef >= 0 else -1)))

# Print results for inspection
print(f"n={len(sub)}")
print(model.summary())
print(f"coef_is_human={coef}")
print(f"se_is_human={se}")
print(f"odds_ratio_is_human={or_val}")
print(f"pval_is_human={pval}")
print(f"scalar={scalar}")
