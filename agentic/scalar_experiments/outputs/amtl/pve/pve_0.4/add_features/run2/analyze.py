import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# Load data
path = 'amtl.csv'
df = pd.read_csv(path)

# Keep relevant columns and drop missing
cols = ['num_amtl', 'age', 'prob_male', 'tooth_class', 'genus']
df = df[cols].dropna().copy()

# Fit linear model controlling for age, sex, tooth class
formula = 'num_amtl ~ age + prob_male + C(tooth_class) + C(genus, Treatment(reference="Homo sapiens"))'
model = smf.ols(formula, data=df).fit()

# Extract genus contrasts vs Homo sapiens
coef = model.params
pvals = model.pvalues

contrasts = {}
for g in ['Pan', 'Pongo', 'Papio']:
    term = f'C(genus, Treatment(reference="Homo sapiens"))[T.{g}]'
    if term in coef:
        contrasts[g] = {
            'coef_vs_homo': float(coef[term]),
            'p_value': float(pvals[term])
        }

# Compute adjusted marginal means for each genus by averaging predictions over covariates
marginal_means = {}
for g in ['Homo sapiens', 'Pan', 'Pongo', 'Papio']:
    df_g = df.copy()
    df_g['genus'] = g
    preds = model.predict(df_g)
    marginal_means[g] = float(preds.mean())

# Save results for inspection
results = {
    'n': int(len(df)),
    'contrasts_vs_homo': contrasts,
    'marginal_means': marginal_means,
    'model_r2': float(model.rsquared),
    'model_pvalues': {k: float(v) for k, v in pvals.items()}
}

with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
