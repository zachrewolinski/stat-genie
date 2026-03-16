import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm


df = pd.read_csv('amtl.csv')

# Ensure categorical
for col in ['feature1', 'feature8']:
    df[col] = df[col].astype('category')

# OLS with robust SEs; genus reference is Homo sapiens
formula = 'feature3 ~ C(feature8, Treatment(reference="Homo sapiens")) + feature5 + feature7 + C(feature1)'
model = smf.ols(formula, data=df).fit(cov_type='HC3')

print(model.summary())

# Extract coefficients for genus comparisons
coef = model.params
se = model.bse
pvals = model.pvalues

# Collect comparisons vs Homo sapiens
comparisons = {}
for genus in ['Pan', 'Papio', 'Pongo']:
    term = f'C(feature8, Treatment(reference="Homo sapiens"))[T.{genus}]'
    if term in coef:
        comparisons[genus] = {
            'coef': float(coef[term]),
            'se': float(se[term]),
            'pvalue': float(pvals[term]),
        }

print('\nComparisons vs Homo sapiens:')
for genus, stats in comparisons.items():
    print(genus, stats)

# G-computation adjusted means: predict for each genus using observed covariates
pred_means = {}
for genus in ['Homo sapiens', 'Pan', 'Papio', 'Pongo']:
    df_tmp = df.copy()
    df_tmp['feature8'] = genus
    pred = model.predict(df_tmp)
    pred_means[genus] = float(pred.mean())

print('\nAdjusted means (g-computation, same covariate distribution):')
for k, v in pred_means.items():
    print(k, v)

# Difference Homo sapiens vs average of non-human (weighted by sample sizes of non-human genera)
nonhuman = ['Pan', 'Papio', 'Pongo']
weights = {g: (df['feature8'] == g).mean() for g in nonhuman}
# normalize weights within nonhuman
w_sum = sum(weights.values())
weights = {g: w / w_sum for g, w in weights.items()}
nonhuman_mean = sum(pred_means[g] * weights[g] for g in nonhuman)

diff = pred_means['Homo sapiens'] - nonhuman_mean
print('\nWeighted nonhuman mean:', nonhuman_mean)
print('Homo sapiens minus nonhuman mean:', diff)

# Test contrast: Homo sapiens vs weighted nonhuman mean
# Build contrast vector for linear combination of params
# model params include intercept, genus indicators for non-human (Pan, Papio, Pongo), age, sex, tooth class indicators.
# Homo sapiens predicted mean uses intercept + covariates + tooth class indicators.
# Nonhuman mean uses intercept + weighted genus indicator effects + same covariates.
# Thus contrast is negative weighted sum of genus indicator effects.
contrast = np.zeros(len(model.params))
param_names = model.params.index.tolist()
for genus in nonhuman:
    term = f'C(feature8, Treatment(reference="Homo sapiens"))[T.{genus}]'
    if term in param_names:
        idx = param_names.index(term)
        contrast[idx] = -weights[genus]

# t-test for difference between Homo sapiens and weighted nonhuman mean
# This tests whether Homo sapiens is higher (positive diff) by testing contrast < 0 for others; here linear combination is Homo - nonhuman.
# We compute t-test for contrast = 0, and sign of diff indicates direction.
res = model.t_test(contrast)
print('\nContrast test (Homo sapiens minus weighted nonhuman mean):')
print('t', float(res.tvalue), 'p', float(res.pvalue))
