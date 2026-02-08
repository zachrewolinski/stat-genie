import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
raw = pd.read_csv('amtl.csv')

# Map shuffled columns to semantic names
# Based on value ranges and metadata
# sockets: tooth class (Anterior/Posterior/Premolar)
# prob_male: specimen id (unused)
# genus: integer count of missing teeth for class
# age: integer count of observable sockets for class
# pop: estimated age at death (years)
# num_amtl: uncertainty of age (unused)
# stdev_age: probability male (sex estimate)
# tooth_class: genus (Homo sapiens, Pan, Papio, Pongo)
# specimen: region (unused)

df = raw.rename(
    columns={
        'sockets': 'tooth_class',
        'prob_male': 'specimen_id',
        'genus': 'num_missing',
        'age': 'num_observed',
        'pop': 'age_at_death',
        'num_amtl': 'age_uncertainty',
        'stdev_age': 'prob_male',
        'tooth_class': 'genus',
        'specimen': 'region',
    }
)

# Drop invalid rows where missing > observed
valid = df['num_missing'] <= df['num_observed']

df = df.loc[valid].copy()

# Build binomial GLM with counts
# Response: num_missing out of num_observed
# Predictors: genus, age_at_death, prob_male, tooth_class

# Use proportion with frequency weights
# Add small jitter to prob_male to avoid perfect separation when 0/1?

formula = 'missing_rate ~ C(genus) + age_at_death + prob_male + C(tooth_class)'

df['missing_rate'] = df['num_missing'] / df['num_observed']

model = smf.glm(
    formula=formula,
    data=df,
    family=sm.families.Binomial(),
    freq_weights=df['num_observed']
).fit()

# Marginal predicted missing rates for each genus at observed covariates

genera = df['genus'].unique()

preds = {}
for g in genera:
    temp = df.copy()
    temp['genus'] = g
    preds[g] = model.predict(temp).mean()

# Compute human vs non-human average
human_rate = preds.get('Homo sapiens')
nonhuman_rates = [preds[g] for g in preds if g != 'Homo sapiens']
nonhuman_avg = float(np.mean(nonhuman_rates)) if nonhuman_rates else np.nan

# Difference in percentage points
if human_rate is None or np.isnan(nonhuman_avg):
    diff = np.nan
else:
    diff = human_rate - nonhuman_avg

# For a rough strength measure, compute z for Homo sapiens vs average of non-human using contrasts
# Compute predicted rates by genus is sufficient for scalar decision.

print('Rows used:', len(df))
print('Model converged:', model.converged)
print('Predicted missing rate by genus:', preds)
print('Human rate:', human_rate)
print('Non-human avg rate:', nonhuman_avg)
print('Difference (human - nonhuman):', diff)

# Write a simple json-like summary for manual inspection
with open('analysis_summary.txt', 'w') as f:
    f.write(f'rows_used={len(df)}\n')
    f.write(f'human_rate={human_rate}\n')
    f.write(f'nonhuman_avg_rate={nonhuman_avg}\n')
    f.write(f'diff={diff}\n')
    f.write('rates_by_genus=' + str(preds) + '\n')
    f.write(str(model.summary()))
