import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats

# Load data
df = pd.read_csv('amtl.csv')

# Map columns to conceptual variables based on inspection
# sockets -> tooth class (Anterior/Posterior/Premolar)
# tooth_class -> genus group (Homo sapiens, Pan, Papio, Pongo)
# pop -> age at death (continuous)
# stdev_age -> prob male (0-1)
# age -> number of observable sockets for that tooth class
# genus -> log of missing count (AMTL) for that tooth class

# Construct derived variables
missing_count = np.exp(df['genus'])
# cap to observable sockets if needed (rare numeric imprecision)
missing_count = np.minimum(missing_count, df['age'])

df = df.copy()
df['missing_count'] = missing_count

df['missing_prop'] = df['missing_count'] / df['age']

# Rename for clarity (avoid name collisions)
df = df.rename(columns={
    'sockets': 'tooth_class',
    'tooth_class': 'genus_group',
    'pop': 'age_at_death',
    'stdev_age': 'prob_male',
    'age': 'observable_sockets',
    'prob_male': 'specimen_id',
    'specimen': 'population'
})

# Ensure categorical
for col in ['tooth_class', 'genus_group']:
    df[col] = df[col].astype('category')

# GLM binomial with weights = observable sockets
formula = 'missing_prop ~ C(genus_group, Treatment(reference="Homo sapiens")) + age_at_death + prob_male + C(tooth_class)'
model = smf.glm(formula=formula, data=df, family=sm.families.Binomial(), freq_weights=df['observable_sockets'])
res = model.fit()

print('Model fitted. Summary:')
print(res.summary())

# Average marginal predicted missing proportion for each genus group

genus_levels = list(df['genus_group'].cat.categories)

avg_preds = {}
for g in genus_levels:
    df_tmp = df.copy()
    df_tmp['genus_group'] = g
    preds = res.predict(df_tmp)
    avg_pred = np.average(preds, weights=df_tmp['observable_sockets'])
    avg_preds[g] = avg_pred

print('\nAverage predicted missing proportion by genus group (age/sex/tooth class adjusted):')
for g, p in avg_preds.items():
    print(f'{g}: {p:.4f}')

# Pairwise contrasts: each non-human vs Homo (reference)
print('\nPairwise contrasts vs Homo sapiens (log-odds differences):')
params = res.params
bse = res.bse

# Extract coefficients for each non-human
for g in genus_levels:
    if g == 'Homo sapiens':
        continue
    term = f'C(genus_group, Treatment(reference="Homo sapiens"))[T.{g}]'
    if term in params.index:
        coef = params[term]
        se = bse[term]
        z = coef / se
        p = 2 * (1 - stats.norm.cdf(abs(z)))
        print(f'{g} vs Homo: coef={coef:.4f}, z={z:.2f}, p={p:.4g}')
    else:
        print(f'Missing term for {g}')

# Save key outputs for later use
summary = {
    'avg_preds': avg_preds,
    'coef': params.to_dict(),
    'pvalues': res.pvalues.to_dict(),
    'n_rows': len(df),
    'n_specimens': df["specimen_id"].nunique() if 'specimen_id' in df.columns else None
}

import json
with open('analysis_results.json', 'w') as f:
    json.dump(summary, f, indent=2)
