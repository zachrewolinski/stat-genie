import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

# Load data
info = json.load(open('info.json'))
df = pd.read_csv('amtl.csv')

# Basic cleaning: keep rows with required fields
req_cols = ['num_amtl', 'sockets', 'age', 'prob_male', 'tooth_class', 'genus']
df = df.dropna(subset=req_cols).copy()

# Filter to valid sockets > 0
# Also ensure num_amtl between 0 and sockets
mask = (df['sockets'] > 0) & (df['num_amtl'] >= 0) & (df['num_amtl'] <= df['sockets'])
df = df.loc[mask].copy()

# Binary indicator for human
# Compare Homo sapiens vs non-human genera aggregated
# This matches the research question's human vs non-human comparison

df['is_human'] = (df['genus'] == 'Homo sapiens').astype(int)

# Build GLM Binomial using counts (successes, failures)
# Include covariates: age, prob_male, tooth_class (categorical)
# Use logit link (default)

# Create failures
failures = df['sockets'] - df['num_amtl']
endog = np.column_stack([df['num_amtl'], failures])

# Use formula with categorical tooth_class
# Add standardized age to improve convergence
# Keep raw age for interpretability? We'll standardize to stabilize
age_mean = df['age'].mean()
age_std = df['age'].std(ddof=0)
if age_std == 0:
    df['age_z'] = 0.0
else:
    df['age_z'] = (df['age'] - age_mean) / age_std

# Fit model
model = sm.GLM(
    endog,
    sm.add_constant(pd.get_dummies(df[['is_human', 'age_z', 'prob_male', 'tooth_class']], columns=['tooth_class'], drop_first=True)),
    family=sm.families.Binomial()
)
result = model.fit()

# Extract human effect
params = result.params
bse = result.bse

coef = params.get('is_human', np.nan)
se = bse.get('is_human', np.nan)

# Wald z and p-value
z = coef / se if se and not np.isnan(se) else np.nan
p = 2 * (1 - stats.norm.cdf(abs(z))) if not np.isnan(z) else np.nan

# Odds ratio and 95% CI
or_val = float(np.exp(coef)) if not np.isnan(coef) else np.nan
ci_low = float(np.exp(coef - 1.96 * se)) if not np.isnan(coef) else np.nan
ci_high = float(np.exp(coef + 1.96 * se)) if not np.isnan(coef) else np.nan

# Predicted AMTL probability for typical covariates
# Use median values for covariates and average tooth_class distribution
# Compute predicted probability for human vs non-human, holding other covariates at mean

def pred_prob(is_human_value):
    # Build a single-row design matrix aligned with model
    # Use mean age_z and mean prob_male, and reference tooth_class category (drop_first)
    row = {
        'const': 1.0,
        'is_human': is_human_value,
        'age_z': 0.0,
        'prob_male': df['prob_male'].mean(),
    }
    # Determine dummy columns in model
    for col in result.model.exog_names:
        if col.startswith('tooth_class_'):
            row[col] = 0.0
    # Ensure order
    x = np.array([row.get(col, 0.0) for col in result.model.exog_names])
    lin = float(np.dot(x, params))
    return 1 / (1 + np.exp(-lin))

p_nonhuman = pred_prob(0)
p_human = pred_prob(1)

diff = p_human - p_nonhuman

# Map evidence to Likert scale
# Heuristic: sign from coef, strength from p-value and effect size (odds ratio)
# Strong positive if OR >> 1 with p < 0.01
# Strong negative if OR << 1 with p < 0.01
# Otherwise moderate/weak

score = 0
if not np.isnan(coef):
    if p < 0.001:
        strength = 90
    elif p < 0.01:
        strength = 75
    elif p < 0.05:
        strength = 60
    elif p < 0.1:
        strength = 40
    else:
        strength = 20

    # Adjust by effect size
    if or_val >= 2.0:
        strength += 10
    elif or_val <= 0.5:
        strength += 10

    if coef > 0:
        score = min(100, int(round(strength)))
    elif coef < 0:
        score = max(-100, -int(round(strength)))
    else:
        score = 0
else:
    score = 0

# Store results for inspection
summary = {
    'n': int(df.shape[0]),
    'coef_is_human': float(coef) if not np.isnan(coef) else None,
    'se_is_human': float(se) if not np.isnan(se) else None,
    'z_is_human': float(z) if not np.isnan(z) else None,
    'p_is_human': float(p) if not np.isnan(p) else None,
    'or_is_human': or_val,
    'or_ci_low': ci_low,
    'or_ci_high': ci_high,
    'pred_p_human': float(p_human),
    'pred_p_nonhuman': float(p_nonhuman),
    'pred_diff': float(diff),
    'likert_score': int(score),
}

pd.DataFrame([summary]).to_csv('analysis_summary.csv', index=False)

# Write conclusion
with open('conclusion.txt', 'w') as f:
    f.write(str(int(score)))
