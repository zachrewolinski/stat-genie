import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.stats import t as tdist

# Load data
amtl = pd.read_csv('amtl.csv')

# Ensure categorical types
amtl['genus'] = amtl['genus'].astype('category')
amtl['tooth_class'] = amtl['tooth_class'].astype('category')

# Set Homo sapiens as reference by reordering categories
cats = ['Homo sapiens', 'Pan', 'Papio', 'Pongo']
cats = [c for c in cats if c in amtl['genus'].cat.categories]
amtl['genus'] = amtl['genus'].cat.reorder_categories(cats, ordered=False)

# Fit OLS with cluster-robust SE by specimen
formula = 'num_amtl ~ C(genus) + age + prob_male + C(tooth_class) + sockets'
model = smf.ols(formula, data=amtl).fit(cov_type='cluster', cov_kwds={'groups': amtl['specimen']})

print(model.summary())

# Extract coefficients for genus (relative to Homo sapiens)
params = model.params
bse = model.bse
pvalues = model.pvalues

# Pairwise contrasts: Homo - other genus (since reference is Homo)
results = {}
for g in ['Pan', 'Papio', 'Pongo']:
    key = f'C(genus)[T.{g}]'
    if key in params:
        diff = -params[key]
        se = bse[key]
        t_stat = diff / se
        df = model.df_resid
        p = 2 * (1 - tdist.cdf(abs(t_stat), df))
        results[g] = {'homo_minus_g': diff, 'se': se, 't': t_stat, 'p': p}

print('Pairwise Homo - genus differences:')
for g, r in results.items():
    print(g, r)

# Marginal means by genus: average predictions across observed covariates

def marginal_mean(genus):
    df = amtl.copy()
    df['genus'] = genus
    pred = model.predict(df)
    return float(pred.mean())

marginal = {g: marginal_mean(g) for g in amtl['genus'].cat.categories}
print('Marginal mean num_amtl by genus:', marginal)

# Save results
import json
with open('model_results.json','w') as f:
    json.dump({'params': params.to_dict(), 'pvalues': pvalues.to_dict(), 'pairwise': results, 'marginal': marginal}, f, indent=2)

