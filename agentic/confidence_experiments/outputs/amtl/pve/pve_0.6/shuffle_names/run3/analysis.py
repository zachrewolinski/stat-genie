import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('amtl.csv')

# Map columns to conceptual variables
# sockets -> tooth_class (Anterior/Posterior/Premolar)
# tooth_class -> genus (Homo sapiens, Pan, Papio, Pongo)
# prob_male -> specimen ID
# specimen -> population
# stdev_age -> prob_male (sex)
# pop -> age at death
# age -> number of observable sockets per tooth class
# genus -> number of AMTL per tooth class (possibly transformed)

# Construct AMTL rate (per observable sockets)

# To avoid division by zero (none expected)
rate = df['genus'] / df['age']

df = df.copy()
df['amtl_rate'] = rate

# Basic cleaning: exclude any rows with non-positive sockets count (shouldn't exist)
df = df[df['age'] > 0]

# Fit linear model with robust SEs
# Predictors: genus category, age at death, sex probability, tooth class

model = smf.ols(
    'amtl_rate ~ C(tooth_class) + pop + stdev_age + C(sockets)',
    data=df
).fit(cov_type='HC3')

print(model.summary())

# Extract coefficient for Homo sapiens relative to reference category
# By default, statsmodels uses first category alphabetically as reference.
# We relevel so that non-human genus is reference? Instead, we compute
# estimated mean difference between Homo sapiens and average non-human.

# Relevel with Homo sapiens as reference to get direct contrasts
model_h = smf.ols(
    'amtl_rate ~ C(tooth_class, Treatment(reference="Homo sapiens")) + pop + stdev_age + C(sockets)',
    data=df
).fit(cov_type='HC3')

print('\nHomo sapiens reference model:')
print(model_h.summary())

# Extract coefficients for non-human vs Homo
coef = model_h.params
se = model_h.bse
pvals = model_h.pvalues

# gather contrasts
contrasts = {k: (coef[k], se[k], pvals[k]) for k in coef.index if k.startswith('C(tooth_class')[0:]}  # placeholder

# We'll compute contrasts for each genus vs Homo

contrasts = {}
for k in coef.index:
    if k.startswith('C(tooth_class')[0:]:
        pass

# Actually filter properly
for k in coef.index:
    if k.startswith('C(tooth_class'):
        contrasts[k] = (coef[k], se[k], pvals[k])

print('\nContrasts vs Homo sapiens:')
for k, (c, s, p) in contrasts.items():
    print(k, 'coef', c, 'se', s, 'p', p)

# Compute estimated marginal means for Homo sapiens vs non-human average
# by predicting at observed covariates, then averaging by genus

pred = model_h.predict(df)

df_pred = df[['tooth_class']].copy()
df_pred['pred'] = pred

mean_by_genus = df_pred.groupby('tooth_class')['pred'].mean()
print('\nPredicted mean AMTL rate by genus:')
print(mean_by_genus)

# Difference Homo vs non-human average
nonhuman_mean = mean_by_genus.drop('Homo sapiens').mean()
human_mean = mean_by_genus.loc['Homo sapiens']
print('\nHomo mean', human_mean, 'non-human mean', nonhuman_mean, 'diff', human_mean - nonhuman_mean)

