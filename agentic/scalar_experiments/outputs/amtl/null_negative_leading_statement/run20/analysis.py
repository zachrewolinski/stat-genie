import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('amtl.csv')

# Basic cleaning
# Ensure numeric columns
for col in ['num_amtl','sockets','age','prob_male']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with missing essentials
needed = ['num_amtl','sockets','age','prob_male','tooth_class','genus']
df = df.dropna(subset=needed).copy()

# Create proportion and human indicator
# Homo sapiens labeled exactly

df['is_human'] = (df['genus'] == 'Homo sapiens').astype(int)

df['amtl_prop'] = df['num_amtl'] / df['sockets']

# Fit binomial GLM with weights = sockets
model = smf.glm(
    'amtl_prop ~ is_human + age + prob_male + C(tooth_class)',
    data=df,
    family=sm.families.Binomial(),
    freq_weights=df['sockets']
)

res = model.fit(cov_type='HC1')

coef = res.params['is_human']
se = res.bse['is_human']
pval = res.pvalues['is_human']
ci_low, ci_high = res.conf_int().loc['is_human']

# Predicted probability difference (average across observed covariates)

# Use full data and toggle is_human

df0 = df.copy()
df0['is_human'] = 0

df1 = df.copy()
df1['is_human'] = 1

pred0 = res.predict(df0)
pred1 = res.predict(df1)

# Weighted mean using sockets
weights = df['sockets'].to_numpy()

diff = np.average(pred1 - pred0, weights=weights)

# Basic descriptive rates by genus
rate_by_genus = (
    df.groupby('genus').apply(lambda g: g['num_amtl'].sum() / g['sockets'].sum())
).sort_values(ascending=False)

print('N rows:', len(df))
print('Human count:', df['is_human'].sum())
print('is_human coef (log-odds):', coef)
print('SE:', se)
print('p-value:', pval)
print('95% CI:', (ci_low, ci_high))
print('Avg predicted prob diff (human - nonhuman):', diff)
print('Rate by genus (AMTL per socket):')
print(rate_by_genus)

# Save outputs for later use
out = {
    'coef': float(coef),
    'se': float(se),
    'pval': float(pval),
    'ci_low': float(ci_low),
    'ci_high': float(ci_high),
    'avg_pred_diff': float(diff),
    'rate_by_genus': rate_by_genus.to_dict(),
    'n': int(len(df)),
    'human_n': int(df['is_human'].sum())
}

with open('analysis_results.json', 'w') as f:
    json.dump(out, f, indent=2)
