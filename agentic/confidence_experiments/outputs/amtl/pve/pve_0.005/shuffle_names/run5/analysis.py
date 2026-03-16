import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('amtl.csv')

# Map variables based on inspection
_df = _df.rename(columns={
    'genus': 'missing_count',
    'age': 'sockets_count',
    'pop': 'age_at_death',
    'stdev_age': 'prob_male',
    'sockets': 'tooth_class',
    'tooth_class': 'genus',
    'prob_male': 'specimen_id',
})

# Compute AMTL rate per sockets
_df['amtl_rate'] = _df['missing_count'] / _df['sockets_count']

# Set genus baseline to Homo sapiens for contrasts
_genus_order = ['Homo sapiens', 'Pan', 'Papio', 'Pongo']
_df['genus'] = pd.Categorical(_df['genus'], categories=_genus_order, ordered=True)

# Weighted least squares with cluster-robust SE by specimen
model = smf.wls(
    'amtl_rate ~ C(genus) + age_at_death + prob_male + C(tooth_class)',
    data=_df,
    weights=_df['sockets_count']
)
results = model.fit(cov_type='cluster', cov_kwds={'groups': _df['specimen_id']})

# Extract coefficients for non-human genera (difference vs Homo sapiens)
coef = results.params
pvals = results.pvalues

# Pairwise differences: Pan - Homo, Papio - Homo, Pongo - Homo
comparisons = {}
for g in ['Pan', 'Papio', 'Pongo']:
    term = f'C(genus)[T.{g}]'
    if term in coef:
        comparisons[g] = {
            'diff_vs_homo': coef[term],
            'pvalue': pvals[term]
        }

# Adjusted mean AMTL rate for each genus at mean covariates and averaged tooth_class
# Use model prediction on a balanced grid of tooth_class
mean_age = _df['age_at_death'].mean()
mean_sex = _df['prob_male'].mean()

adj_means = {}
for g in _genus_order:
    # average over tooth classes equally
    preds = []
    for tc in _df['tooth_class'].unique():
        row = pd.DataFrame({
            'genus': [g],
            'age_at_death': [mean_age],
            'prob_male': [mean_sex],
            'tooth_class': [tc]
        })
        preds.append(results.predict(row)[0])
    adj_means[g] = float(np.mean(preds))

# Save summary outputs for inspection
summary = {
    'comparisons': comparisons,
    'adj_means': adj_means,
    'model_r2': float(results.rsquared),
    'n_obs': int(results.nobs)
}

import json
with open('analysis_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(results.summary())
print(json.dumps(summary, indent=2))
