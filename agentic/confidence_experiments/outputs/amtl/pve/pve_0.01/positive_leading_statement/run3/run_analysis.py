import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('amtl.csv')

# Basic checks
missing = df.isna().sum()
print('Missing values by column:\n', missing)

# Create Homo indicator

df['homo'] = (df['genus'] == 'Homo sapiens').astype(int)

# OLS with robust SE
model = smf.ols('num_amtl ~ homo + age + prob_male + C(tooth_class)', data=df).fit(cov_type='HC3')
print(model.summary())

# Extract homo effect
coef = model.params['homo']
se = model.bse['homo']
pval = model.pvalues['homo']
ci_low, ci_high = model.conf_int().loc['homo']
print('homo coef', coef, 'SE', se, 'p', pval, 'CI', (ci_low, ci_high))

# Adjusted predicted difference at mean covariates
mean_age = df['age'].mean()
mean_prob_male = df['prob_male'].mean()
# Use most common tooth class for reference
mode_tooth = df['tooth_class'].mode().iloc[0]

pred_df = pd.DataFrame({
    'homo': [0, 1],
    'age': [mean_age, mean_age],
    'prob_male': [mean_prob_male, mean_prob_male],
    'tooth_class': [mode_tooth, mode_tooth]
})

preds = model.predict(pred_df)
print('predicted num_amtl at mean covariates (non-homo, homo):', preds.tolist())

# Model with genus categories to compare each non-human genus
model_genus = smf.ols('num_amtl ~ C(genus) + age + prob_male + C(tooth_class)', data=df).fit(cov_type='HC3')
print(model_genus.summary())

# Output key results as JSON for quick parsing
out = {
    'homo_coef': float(coef),
    'homo_se': float(se),
    'homo_p': float(pval),
    'homo_ci': [float(ci_low), float(ci_high)],
    'pred_non_homo': float(preds.iloc[0]),
    'pred_homo': float(preds.iloc[1]),
}
print('RESULT_JSON', json.dumps(out))
