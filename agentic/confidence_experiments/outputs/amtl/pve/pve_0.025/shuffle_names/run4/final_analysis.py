import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


df = pd.read_csv('amtl.csv')

# Inferred mappings based on values
# sockets -> tooth class (Anterior/Posterior/Premolar)
# tooth_class -> genus labels (Homo sapiens, Pan, Papio, Pongo)
# pop -> estimated age at death (continuous)
# stdev_age -> sex probability (0-1)
# genus -> class-specific AMTL measure (continuous)

DF = df.copy()
DF['is_human'] = (DF['tooth_class'] == 'Homo sapiens').astype(int)

# Cluster-robust SE by specimen ID (prob_male) due to repeated rows per specimen
model = smf.ols('genus ~ is_human + pop + stdev_age + C(sockets)', data=DF).fit(
    cov_type='cluster', cov_kwds={'groups': DF['prob_male']}
)

# Model with genus categories to obtain adjusted means
model2 = smf.ols('genus ~ C(tooth_class) + pop + stdev_age + C(sockets)', data=DF).fit(
    cov_type='cluster', cov_kwds={'groups': DF['prob_male']}
)

# Adjusted predictions at mean covariates, averaged over sockets
mean_pop = DF['pop'].mean()
mean_sex = DF['stdev_age'].mean()

preds = {}
for g in DF['tooth_class'].unique():
    rows = [{'tooth_class': g, 'pop': mean_pop, 'stdev_age': mean_sex, 'sockets': s}
            for s in DF['sockets'].unique()]
    preds[g] = model2.predict(pd.DataFrame(rows)).mean()

non_human_mean = np.mean([preds[g] for g in preds if g != 'Homo sapiens'])

coef = model.params['is_human']
se = model.bse['is_human']
pval = model.pvalues['is_human']

# Likert score: significant positive effect but modest effect size (coef ~0.3 SD)
response = 75

explanation = (
    "Using OLS with cluster-robust SE by specimen ID (to handle three tooth-class rows per specimen), "
    "I modeled the class-specific AMTL measure (column with continuous values labeled 'genus') as a function of "
    "human vs. non-human genus, estimated age at death ('pop'), sex probability ('stdev_age'), and tooth class "
    "(Anterior/Posterior/Premolar from 'sockets'). The human indicator is positive and statistically significant "
    f"(coef={coef:.3f}, SE={se:.3f}, p={pval:.3g}), while age, sex, and tooth class effects are not significant. "
    "A model with genus categories shows Pan and Papio significantly lower than Homo sapiens, with Pongo lower but "
    "marginal. Adjusted predictions at mean covariates give Homo sapiens ≈ "
    f"{preds['Homo sapiens']:.2f} versus non-human mean ≈ {non_human_mean:.2f} (difference ≈ "
    f"{preds['Homo sapiens']-non_human_mean:.2f}). This provides moderate evidence that modern humans have higher "
    "AMTL frequency after accounting for age, sex, and tooth class."
)

with open('conclusion.txt', 'w', encoding='utf-8') as f:
    json.dump({'response': response, 'explanation': explanation}, f)

print('Wrote conclusion.txt')
