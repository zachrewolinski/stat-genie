import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
file_path = 'amtl.csv'
df = pd.read_csv(file_path)

# Basic cleaning
# Ensure categorical types
df['tooth_class'] = df['tooth_class'].astype('category')
df['genus'] = df['genus'].astype('category')

# Create binary indicator for Homo sapiens
df['is_homo'] = (df['genus'] == 'Homo sapiens').astype(int)

# Drop rows with missing values in key fields
key_cols = ['num_amtl', 'sockets', 'age', 'prob_male', 'tooth_class', 'is_homo', 'specimen']
df_model = df.dropna(subset=key_cols).copy()

# Fit OLS with cluster-robust SE by specimen (repeated measures across tooth classes)
formula = 'num_amtl ~ is_homo + age + prob_male + sockets + C(tooth_class)'
model = smf.ols(formula=formula, data=df_model).fit(cov_type='cluster', cov_kwds={'groups': df_model['specimen']})

# Also fit with genus categories to see each non-human genus vs Homo for context
model_genus = smf.ols('num_amtl ~ C(genus) + age + prob_male + sockets + C(tooth_class)', data=df_model).fit(
    cov_type='cluster', cov_kwds={'groups': df_model['specimen']}
)

# Compute adjusted mean difference (Homo vs nonhuman) using model predictions
# Predict for each row twice: is_homo=1 and is_homo=0, keep other covariates fixed
pred_homo = model.predict(df_model.assign(is_homo=1))
pred_nonhomo = model.predict(df_model.assign(is_homo=0))
adj_diff = (pred_homo - pred_nonhomo).mean()

# Summary stats by genus
group_means = df_model.groupby('genus')['num_amtl'].agg(['mean','std','count']).reset_index()

results = {
    'n_rows': len(df_model),
    'n_specimens': df_model['specimen'].nunique(),
    'coef_is_homo': model.params['is_homo'],
    'se_is_homo': model.bse['is_homo'],
    'p_is_homo': model.pvalues['is_homo'],
    'adj_diff_homo_vs_nonhomo': adj_diff,
    'model_r2': model.rsquared,
    'model_genus_params': model_genus.params.to_dict(),
    'model_genus_pvalues': model_genus.pvalues.to_dict(),
    'group_means': group_means.to_dict(orient='records')
}

# Save results to a JSON file for inspection
import json
with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
