import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


df = pd.read_csv('amtl.csv')

# Derive AMTL rate per observable sockets
# Note: num_amtl is not integer (likely noise-added), so treat as continuous.
df['amtl_rate'] = df['num_amtl'] / df['sockets']

# Binary indicator for modern humans
df['is_human'] = (df['genus'] == 'Homo sapiens').astype(int)

# Fit weighted least squares with robust SEs
formula = 'amtl_rate ~ is_human + age + prob_male + C(tooth_class)'
model = smf.wls(formula, data=df, weights=df['sockets']).fit(cov_type='HC3')

# Extract effect for humans
coef = model.params['is_human']
se = model.bse['is_human']
pval = model.pvalues['is_human']

# Adjusted mean difference at average covariates
avg_age = df['age'].mean()
avg_male = df['prob_male'].mean()
# Use reference tooth class (first category in patsy), so compute predictions at that reference
ref_tooth = df['tooth_class'].astype('category').cat.categories[0]

pred_df = pd.DataFrame({
    'is_human': [0, 1],
    'age': [avg_age, avg_age],
    'prob_male': [avg_male, avg_male],
    'tooth_class': [ref_tooth, ref_tooth],
})

pred = model.predict(pred_df)
adj_diff = float(pred.iloc[1] - pred.iloc[0])

# Raw group means for context
means = df.groupby('is_human')['amtl_rate'].mean()

out = {
    'coef_is_human': float(coef),
    'se_is_human': float(se),
    'pval_is_human': float(pval),
    'adj_diff_rate': float(adj_diff),
    'mean_rate_nonhuman': float(means.loc[0]),
    'mean_rate_human': float(means.loc[1]),
    'n': int(len(df)),
}

with open('analysis_results.json', 'w') as f:
    json.dump(out, f, indent=2)

print(json.dumps(out, indent=2))
