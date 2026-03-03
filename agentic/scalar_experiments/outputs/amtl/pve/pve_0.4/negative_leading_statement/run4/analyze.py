import json
import pandas as pd
import statsmodels.formula.api as smf
import numpy as np

# Load data
path = 'amtl.csv'
df = pd.read_csv(path)

# Prepare variables
# binary indicator for Homo sapiens
DF_HUMAN_LABEL = 'Homo sapiens'
df['is_human'] = (df['genus'] == DF_HUMAN_LABEL).astype(int)

# Ensure categories
for col in ['tooth_class', 'genus', 'specimen']:
    df[col] = df[col].astype('category')

# Model: human vs non-human controlling for age, sex, tooth class
# Use cluster-robust SEs by specimen to account for repeated measures
model = smf.ols('num_amtl ~ is_human + age + prob_male + C(tooth_class)', data=df)
res = model.fit(cov_type='cluster', cov_kwds={'groups': df['specimen']})

coef = res.params['is_human']
se = res.bse['is_human']
ci_low, ci_high = res.conf_int().loc['is_human']
pval = res.pvalues['is_human']

# Cross-check: full genus categories with Homo sapiens as reference
model_genus = smf.ols('num_amtl ~ C(genus, Treatment(reference="Homo sapiens")) + age + prob_male + C(tooth_class)', data=df)
res_genus = model_genus.fit(cov_type='cluster', cov_kwds={'groups': df['specimen']})

# Descriptive stats by genus
means = df.groupby('genus')['num_amtl'].mean().sort_values(ascending=False)
counts = df['genus'].value_counts()

# Save a compact results json for later use
out = {
    'coef_is_human': float(coef),
    'se_is_human': float(se),
    'pval_is_human': float(pval),
    'ci_low': float(ci_low),
    'ci_high': float(ci_high),
    'n': int(df.shape[0]),
    'n_specimens': int(df['specimen'].nunique()),
    'mean_by_genus': means.to_dict(),
    'count_by_genus': counts.to_dict(),
    'genus_params': {k: float(v) for k, v in res_genus.params.items() if k.startswith('C(genus')},
    'genus_pvalues': {k: float(v) for k, v in res_genus.pvalues.items() if k.startswith('C(genus')},
}

with open('analysis_results.json', 'w') as f:
    json.dump(out, f, indent=2)

print(json.dumps(out, indent=2))
