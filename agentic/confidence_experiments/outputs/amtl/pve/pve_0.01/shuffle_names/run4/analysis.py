import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import json

# Load data

df = pd.read_csv('amtl.csv')

# Map columns to conceptual variables based on values
# sockets -> tooth class; tooth_class -> genus; pop -> age at death; stdev_age -> prob_male; age -> sockets count; genus -> AMTL frequency (numeric)
# prob_male column in raw data appears to be specimen id (strings), so rename it accordingly.

df = df.rename(columns={
    'sockets': 'tooth_class',
    'tooth_class': 'genus',
    'pop': 'age_at_death',
    'stdev_age': 'prob_male',
    'age': 'num_sockets',
    'genus': 'amtl_freq',
    'prob_male': 'specimen_id'
})

# Ensure categorical variables
for col in ['tooth_class', 'genus']:
    df[col] = df[col].astype('category')

# OLS on amtl_freq (treated as frequency / logit scale) with covariates
model = smf.ols('amtl_freq ~ C(genus) + age_at_death + prob_male + C(tooth_class)', data=df).fit(cov_type='HC3')

# Compute contrasts: Homo sapiens vs each non-human genus, and vs average non-human
levels = df['genus'].cat.categories.tolist()
ref = levels[0]

mean_age = df['age_at_death'].mean()
mean_prob_male = df['prob_male'].mean()
most_common_tooth = df['tooth_class'].mode()[0]

preds = {}
for g in levels:
    row = pd.DataFrame({
        'genus': [g],
        'age_at_death': [mean_age],
        'prob_male': [mean_prob_male],
        'tooth_class': [most_common_tooth]
    })
    preds[g] = float(model.predict(row)[0])

if 'Homo sapiens' in levels:
    homo = 'Homo sapiens'
else:
    homo = levels[0]

contrasts = {}
for g in levels:
    if g == homo:
        continue
    params = model.params.index.tolist()
    v = np.zeros(len(params))
    if homo != ref:
        v[params.index(f'C(genus)[T.{homo}]')] = 1
    if g != ref:
        v[params.index(f'C(genus)[T.{g}]')] = -1
    ttest = model.t_test(v)
    contrasts[g] = {
        'diff': float(preds[homo] - preds[g]),
        't': float(ttest.tvalue),
        'p': float(ttest.pvalue)
    }

non_humans = [g for g in levels if g != homo]
if non_humans:
    params = model.params.index.tolist()
    v = np.zeros(len(params))
    if homo != ref:
        v[params.index(f'C(genus)[T.{homo}]')] = 1
    for g in non_humans:
        if g != ref:
            v[params.index(f'C(genus)[T.{g}]')] -= 1/len(non_humans)
    ttest = model.t_test(v)
    avg_diff = preds[homo] - float(np.mean([preds[g] for g in non_humans]))
    contrast_avg = {
        'diff': float(avg_diff),
        't': float(ttest.tvalue),
        'p': float(ttest.pvalue)
    }
else:
    contrast_avg = None

results = {
    'model_summary': model.summary().as_text(),
    'preds': preds,
    'contrasts': contrasts,
    'contrast_avg': contrast_avg,
    'ref_genus': ref,
    'most_common_tooth_class': most_common_tooth,
    'mean_age_at_death': mean_age,
    'mean_prob_male': mean_prob_male
}

with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print('done')
