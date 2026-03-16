import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('amtl.csv')

# Keep relevant columns
cols = ['num_amtl', 'age', 'prob_male', 'tooth_class', 'genus', 'sockets']
missing_cols = [c for c in cols if c not in _df.columns]
if missing_cols:
    raise ValueError(f"Missing columns: {missing_cols}")

df = _df[cols].copy()

# Drop rows with missing values in analysis variables
analysis_df = df.dropna()

# Indicator for human vs non-human primates
analysis_df['human'] = (analysis_df['genus'] == 'Homo sapiens').astype(int)

# Basic descriptive stats
num_amtl_mean = analysis_df['num_amtl'].mean()
num_amtl_std = analysis_df['num_amtl'].std(ddof=1)
by_genus_mean = analysis_df.groupby('genus')['num_amtl'].mean().sort_values(ascending=False)

# Main model: adjusted difference humans vs non-humans
model = smf.ols('num_amtl ~ human + age + prob_male + C(tooth_class)', data=analysis_df).fit(cov_type='HC3')

coef = model.params['human']
pval = model.pvalues['human']
ci_low, ci_high = model.conf_int().loc['human']

# Secondary model including sockets as a sensitivity check
model_sockets = smf.ols('num_amtl ~ human + age + prob_male + C(tooth_class) + sockets', data=analysis_df).fit(cov_type='HC3')
coef_s = model_sockets.params['human']
pval_s = model_sockets.pvalues['human']
ci_low_s, ci_high_s = model_sockets.conf_int().loc['human']

# Save results for manual write-up
results = {
    'n_rows': int(len(analysis_df)),
    'num_amtl_mean': float(num_amtl_mean),
    'num_amtl_std': float(num_amtl_std),
    'by_genus_mean': by_genus_mean.to_dict(),
    'model_coef_human': float(coef),
    'model_pval_human': float(pval),
    'model_ci_human': [float(ci_low), float(ci_high)],
    'model_sockets_coef_human': float(coef_s),
    'model_sockets_pval_human': float(pval_s),
    'model_sockets_ci_human': [float(ci_low_s), float(ci_high_s)],
}

with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
