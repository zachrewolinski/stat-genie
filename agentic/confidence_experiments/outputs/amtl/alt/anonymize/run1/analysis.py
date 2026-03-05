import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = 'amtl.csv'
df = pd.read_csv(path)

# Rename columns to meaningful names
rename = {
    'feature1': 'tooth_class',
    'feature2': 'specimen_id',
    'feature3': 'missing_teeth',
    'feature4': 'observable_sockets',
    'feature5': 'age',
    'feature6': 'age_uncertainty',
    'feature7': 'sex_estimate',
    'feature8': 'genus',
    'feature9': 'region',
}
df = df.rename(columns=rename)

# Basic cleaning
# Ensure numeric types
for col in ['missing_teeth', 'observable_sockets', 'age', 'age_uncertainty', 'sex_estimate']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with missing key fields or invalid counts
mask_valid = (
    df['missing_teeth'].notna()
    & df['observable_sockets'].notna()
    & df['age'].notna()
    & df['sex_estimate'].notna()
    & df['tooth_class'].notna()
    & df['genus'].notna()
)

df = df.loc[mask_valid].copy()

# Remove rows with impossible counts
# missing_teeth should be between 0 and observable_sockets
invalid = (df['missing_teeth'] < 0) | (df['observable_sockets'] <= 0) | (df['missing_teeth'] > df['observable_sockets'])
if invalid.any():
    df = df.loc[~invalid].copy()

# Create human indicator
# genus values include 'Homo sapiens' for humans
# Non-human includes Pan, Pongo, Papio

df['human'] = (df['genus'] == 'Homo sapiens').astype(int)

# Binomial response: proportion missing with trials = observable_sockets
# Use GLM with binomial and frequency weights

df['prop_missing'] = df['missing_teeth'] / df['observable_sockets']

# Fit model with controls for age, sex, tooth class
formula = 'prop_missing ~ human + age + sex_estimate + C(tooth_class)'
model = smf.glm(formula=formula, data=df, family=sm.families.Binomial(), freq_weights=df['observable_sockets'])
result = model.fit()

# Extract human effect
coef = result.params['human']
se = result.bse['human']
p_value = result.pvalues['human']

# Odds ratio and 95% CI
or_human = np.exp(coef)
ci_low = np.exp(coef - 1.96 * se)
ci_high = np.exp(coef + 1.96 * se)

# Also compute adjusted predicted probabilities for human vs nonhuman at mean covariates
# Use representative values: mean age, mean sex_estimate, and reference tooth class distribution.
# We'll compute marginal predicted probability for each row with human=0/1 and average.

df_pred = df.copy()

df_pred['human'] = 0
pred_nonhuman = result.predict(df_pred)

df_pred['human'] = 1
pred_human = result.predict(df_pred)

# Weighted average probability by sockets (trials)
weights = df['observable_sockets']
mean_nonhuman = np.average(pred_nonhuman, weights=weights)
mean_human = np.average(pred_human, weights=weights)

print('N rows:', len(df))
print('Human rows:', df['human'].sum())
print('Coefficient (human):', coef)
print('SE:', se)
print('p-value:', p_value)
print('Odds ratio:', or_human)
print('95% CI OR:', (ci_low, ci_high))
print('Adjusted mean prob nonhuman:', mean_nonhuman)
print('Adjusted mean prob human:', mean_human)
print('Difference (human - nonhuman):', mean_human - mean_nonhuman)

# Save summary to file for later use
summary = {
    'n_rows': int(len(df)),
    'human_rows': int(df['human'].sum()),
    'coef_human': float(coef),
    'se_human': float(se),
    'p_value_human': float(p_value),
    'odds_ratio_human': float(or_human),
    'or_ci_low': float(ci_low),
    'or_ci_high': float(ci_high),
    'adj_prob_nonhuman': float(mean_nonhuman),
    'adj_prob_human': float(mean_human),
    'adj_diff': float(mean_human - mean_nonhuman)
}

import json
with open('analysis_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
