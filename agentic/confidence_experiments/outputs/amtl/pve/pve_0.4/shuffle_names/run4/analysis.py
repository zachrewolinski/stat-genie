import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy.special import expit

# Load data

df = pd.read_csv('amtl.csv')

# Map columns
# genus (numeric) assumed AMTL-related measure per tooth class
# sockets: tooth class categories
# tooth_class: genus categories
# pop: age at death
# stdev_age: sex probability (prob_male)
# prob_male: specimen id for clustering

# Create indicator for human

df['is_human'] = (df['tooth_class'] == 'Homo sapiens').astype(int)

# Model 1: OLS on genus (AMTL measure)
model1 = smf.ols('genus ~ is_human + pop + stdev_age + C(sockets)', data=df).fit(
    cov_type='cluster', cov_kwds={'groups': df['prob_male']}
)

# Model 2: OLS on expit(genus) as proportion
# This treats genus as logit-scale measure

df['p_amtl'] = expit(df['genus'])
model2 = smf.ols('p_amtl ~ is_human + pop + stdev_age + C(sockets)', data=df).fit(
    cov_type='cluster', cov_kwds={'groups': df['prob_male']}
)

# Model 3: OLS with categorical genus to compare against each non-human genus
model3 = smf.ols('genus ~ C(tooth_class) + pop + stdev_age + C(sockets)', data=df).fit(
    cov_type='cluster', cov_kwds={'groups': df['prob_male']}
)

# Collect results

results = {
    'model1_is_human_coef': model1.params['is_human'],
    'model1_is_human_se': model1.bse['is_human'],
    'model1_is_human_p': model1.pvalues['is_human'],
    'model2_is_human_coef': model2.params['is_human'],
    'model2_is_human_se': model2.bse['is_human'],
    'model2_is_human_p': model2.pvalues['is_human'],
}

# For model3, compute predicted means for each genus category at average covariates
avg_pop = df['pop'].mean()
avg_stdev = df['stdev_age'].mean()
# use reference sockets category (first alphabetic)
ref_socket = sorted(df['sockets'].unique())[0]

preds = {}
for genus in df['tooth_class'].unique():
    tmp = pd.DataFrame({
        'tooth_class': [genus],
        'pop': [avg_pop],
        'stdev_age': [avg_stdev],
        'sockets': [ref_socket],
    })
    preds[genus] = float(model3.predict(tmp)[0])

# Pairwise differences: Homo sapiens vs others (model3)
# Use t_test for contrasts

contrasts = {}

# We'll compute contrasts using design matrix.
# Use Patsy design info to build rows for each genus.
from patsy import dmatrix

# Build design rows for each genus at mean covariates and reference sockets

rows = {}
for genus in df['tooth_class'].unique():
    tmp = pd.DataFrame({
        'tooth_class': [genus],
        'pop': [avg_pop],
        'stdev_age': [avg_stdev],
        'sockets': [ref_socket],
    })
    rows[genus] = dmatrix(model3.model.data.design_info, tmp)

# Compute contrasts Homo vs each other
homo = 'Homo sapiens'
for genus in df['tooth_class'].unique():
    if genus == homo:
        continue
    contrast = rows[homo] - rows[genus]
    ttest = model3.t_test(contrast)
    contrasts[f'Homo_vs_{genus}'] = {
        'diff': float(ttest.effect),
        'pvalue': float(ttest.pvalue),
    }

# Output summary
print('Model1 (genus ~ is_human + covariates):')
print(model1.summary().tables[1])
print('\nModel2 (p_amtl ~ is_human + covariates):')
print(model2.summary().tables[1])
print('\nModel3 (genus ~ genus categories + covariates):')
print(model3.summary().tables[1])
print('\nPredicted genus by genus (avg covariates):')
print(preds)
print('\nContrasts Homo vs others:')
print(contrasts)

# Save results for later use
import json
with open('analysis_results.json', 'w') as f:
    json.dump({'results': results, 'preds': preds, 'contrasts': contrasts}, f, indent=2)
