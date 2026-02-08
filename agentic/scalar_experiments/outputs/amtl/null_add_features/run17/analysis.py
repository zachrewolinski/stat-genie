import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = 'amtl.csv'
df = pd.read_csv(path)

# Basic filtering: keep rows with necessary fields
needed_cols = ['num_amtl','sockets','age','prob_male','tooth_class','genus']
df = df.dropna(subset=needed_cols)

# Create human indicator
# Genus values: include 'Homo sapiens' as human, others as non-human
# Robust to capitalization/spacing
human_mask = df['genus'].astype(str).str.strip().str.lower().eq('homo sapiens')
df = df.loc[df['sockets'] > 0].copy()
df['is_human'] = human_mask.astype(int)

# Ensure numeric types
for col in ['num_amtl','sockets','age','prob_male']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop any remaining missing
model_df = df.dropna(subset=['num_amtl','sockets','age','prob_male','tooth_class','is_human'])

# Binomial GLM with events/total via endog as proportion + weights
model_df['amtl_rate'] = model_df['num_amtl'] / model_df['sockets']

# Use categorical for tooth_class
model_df['tooth_class'] = model_df['tooth_class'].astype('category')

# Fit GLM
formula = 'amtl_rate ~ is_human + age + prob_male + C(tooth_class)'
model = smf.glm(formula=formula, data=model_df, family=sm.families.Binomial(), freq_weights=model_df['sockets'])
result = model.fit()

# Extract human coefficient
coef = result.params.get('is_human', np.nan)
se = result.bse.get('is_human', np.nan)
pval = result.pvalues.get('is_human', np.nan)

# Compute odds ratio
odds_ratio = np.exp(coef) if pd.notnull(coef) else np.nan

# Predict average difference in AMTL rate between human and non-human at mean covariates
mean_age = model_df['age'].mean()
mean_prob_male = model_df['prob_male'].mean()
# reference tooth class: use most frequent category
ref_tooth = model_df['tooth_class'].value_counts().idxmax()

# Construct two rows for prediction
pred_df = pd.DataFrame({
    'is_human': [0, 1],
    'age': [mean_age, mean_age],
    'prob_male': [mean_prob_male, mean_prob_male],
    'tooth_class': [ref_tooth, ref_tooth],
})

pred = result.predict(pred_df)
rate_nonhuman, rate_human = pred.iloc[0], pred.iloc[1]
rate_diff = rate_human - rate_nonhuman

# Save key outputs to json for later inspection
out = {
    'n_rows': int(model_df.shape[0]),
    'coef_is_human': float(coef),
    'se_is_human': float(se),
    'pval_is_human': float(pval),
    'odds_ratio_is_human': float(odds_ratio),
    'pred_rate_nonhuman': float(rate_nonhuman),
    'pred_rate_human': float(rate_human),
    'pred_rate_diff': float(rate_diff),
}

import json
with open('analysis_results.json','w') as f:
    json.dump(out, f, indent=2)

print(json.dumps(out, indent=2))
