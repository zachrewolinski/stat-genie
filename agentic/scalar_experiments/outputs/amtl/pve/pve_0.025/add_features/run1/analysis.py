import json
import pandas as pd
import statsmodels.formula.api as smf

# Load data
df = pd.read_csv('amtl.csv')

# Keep relevant columns
cols = ['num_amtl', 'genus', 'age', 'prob_male', 'tooth_class']
missing = [c for c in cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}")

df = df[cols].dropna().copy()

# Fit linear model with genus, age, sex (prob_male), tooth_class
model = smf.ols('num_amtl ~ C(genus) + age + prob_male + C(tooth_class)', data=df).fit()

# Extract genus coefficients (differences vs baseline)
params = model.params
pvalues = model.pvalues
conf = model.conf_int()

# Identify baseline genus
# statsmodels uses alphabetical baseline by default for C(genus)
# We'll record baseline and the other levels
levels = sorted(df['genus'].unique())
# baseline is first level in alphabetical sort
baseline = levels[0]

# Build adjusted marginal means: predict for each genus by setting genus and averaging
adj_means = {}
for g in levels:
    df_g = df.copy()
    df_g['genus'] = g
    preds = model.predict(df_g)
    adj_means[g] = preds.mean()

# Differences vs Homo sapiens (if present)
comparisons = {}
if 'Homo sapiens' in levels:
    homo = 'Homo sapiens'
    for g in levels:
        if g == homo:
            continue
        # Difference (other - homo) in adjusted means
        diff = adj_means[g] - adj_means[homo]
        comparisons[f'{g} - {homo}'] = diff

# Save results to JSON for inspection
output = {
    'n_rows': len(df),
    'baseline_genus': baseline,
    'genus_levels': levels,
    'adj_means': adj_means,
    'comparisons': comparisons,
    'coefficients': {k: params[k] for k in params.index if k.startswith('C(genus)')},
    'pvalues': {k: pvalues[k] for k in pvalues.index if k.startswith('C(genus)')},
    'conf_int': {k: [conf.loc[k, 0], conf.loc[k, 1]] for k in conf.index if k.startswith('C(genus)')},
    'model_r2': model.rsquared,
}

with open('analysis_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print(json.dumps(output, indent=2))
