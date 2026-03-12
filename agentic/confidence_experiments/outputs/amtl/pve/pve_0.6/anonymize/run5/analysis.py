import pandas as pd
import statsmodels.formula.api as smf
import numpy as np

# Load data
path = 'amtl.csv'
df = pd.read_csv(path)

# Rename for clarity
# feature1: tooth class
# feature2: specimen id
# feature3: AMTL measure (scaled/normalized)
# feature4: observable sockets
# feature5: age
# feature6: age uncertainty
# feature7: sex
# feature8: genus
# feature9: region

# Create Homo sapiens indicator
# In case of any whitespace, normalize

df['genus'] = df['feature8'].astype(str).str.strip()
df['homo'] = (df['genus'] == 'Homo sapiens').astype(int)

# Tooth class categorical
# Use C(feature1) for tooth class

# OLS with cluster-robust SE by specimen
formula = 'feature3 ~ homo + feature5 + feature7 + C(feature1)'
model = smf.ols(formula, data=df)
res = model.fit(cov_type='cluster', cov_kwds={'groups': df['feature2']})

# Also model with full genus categories for comparison
formula_full = 'feature3 ~ C(genus) + feature5 + feature7 + C(feature1)'
model_full = smf.ols(formula_full, data=df)
res_full = model_full.fit(cov_type='cluster', cov_kwds={'groups': df['feature2']})

# Compute group means (adjusted) by predicting at mean age/sex and each genus? We'll do raw means too.
raw_means = df.groupby('genus')['feature3'].mean().sort_values()

# Adjusted means using model_full (setting tooth class distribution maybe). We'll compute average predicted for each genus using observed covariates but genus set.
# This is a standard marginal means approach.

df_base = df.copy()

adjusted_means = {}
for genus in df['genus'].unique():
    df_base['genus'] = genus
    preds = res_full.predict(df_base)
    adjusted_means[genus] = float(np.mean(preds))

# collect results
out = {
    'n_rows': len(df),
    'n_specimens': df['feature2'].nunique(),
    'homo_coef': res.params.get('homo', np.nan),
    'homo_pvalue': res.pvalues.get('homo', np.nan),
    'homo_ci': res.conf_int().loc['homo'].tolist() if 'homo' in res.params else [np.nan, np.nan],
    'homo_se': res.bse.get('homo', np.nan),
    'homo_t': res.tvalues.get('homo', np.nan),
    'raw_means': raw_means.to_dict(),
    'adjusted_means': adjusted_means,
    'model_summary': res.summary().as_text(),
    'model_full_summary': res_full.summary().as_text()
}

# Save to a temp file for inspection
import json
with open('analysis_results.json', 'w') as f:
    json.dump(out, f, indent=2)

print('done')
