import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from patsy import build_design_matrices


df = pd.read_csv('amtl.csv')

# Set categorical order with Homo sapiens as reference

genus_order = ["Homo sapiens", "Pan", "Papio", "Pongo"]

df['genus'] = pd.Categorical(df['genus'], categories=genus_order)

df['tooth_class'] = pd.Categorical(df['tooth_class'])

model = smf.ols('num_amtl ~ C(genus) + age + prob_male + C(tooth_class)', data=df).fit()

# Build marginal means for each genus by averaging predictions over observed covariates

design_info = model.model.data.design_info
cov = model.cov_params()

results = {}

for g in genus_order:
    data_g = df.copy()
    data_g['genus'] = g
    exog_g = build_design_matrices([design_info], data_g)[0]
    xbar = np.asarray(exog_g).mean(axis=0)
    mean_pred = float(xbar @ model.params)
    var = float(xbar @ cov @ xbar.T)
    se = float(np.sqrt(var))
    ci_low = mean_pred - 1.96 * se
    ci_high = mean_pred + 1.96 * se
    results[g] = {
        'mean_pred': mean_pred,
        'se': se,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'xbar': xbar,
    }

# Differences vs Homo sapiens

diffs = {}

xbar_h = results['Homo sapiens']['xbar']

for g in genus_order[1:]:
    xbar_g = results[g]['xbar']
    xdiff = xbar_h - xbar_g
    diff = float(xdiff @ model.params)
    var = float(xdiff @ cov @ xdiff.T)
    se = float(np.sqrt(var))
    ci_low = diff - 1.96 * se
    ci_high = diff + 1.96 * se
    # two-sided p-value
    z = diff / se if se > 0 else np.nan
    from math import erf, sqrt
    # normal cdf
    def norm_cdf(z):
        return 0.5 * (1 + erf(z / sqrt(2)))
    p = 2 * (1 - norm_cdf(abs(z))) if se > 0 else np.nan
    diffs[g] = {
        'diff_h_minus_g': diff,
        'se': se,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'p_value': p,
    }

print('Model summary (coefficients):')
print(model.summary().tables[1])
print('\nMarginal means (adjusted):')
for g, r in results.items():
    print(g, r['mean_pred'], r['ci_low'], r['ci_high'])
print('\nDifferences Homo sapiens - other genera:')
for g, r in diffs.items():
    print(g, r['diff_h_minus_g'], r['ci_low'], r['ci_high'], r['p_value'])

# Save results for downstream use

out = {
    'model_r2': model.rsquared,
    'model_adj_r2': model.rsquared_adj,
    'params': model.params.to_dict(),
    'pvalues': model.pvalues.to_dict(),
    'marginal_means': {k: {kk: v for kk, v in r.items() if kk != 'xbar'} for k, r in results.items()},
    'diffs_vs_homo': diffs,
}

import json
with open('analysis_results.json', 'w') as f:
    json.dump(out, f, indent=2)
