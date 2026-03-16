import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm


df = pd.read_csv('amtl.csv')
# Outcome: AMTL rate (missing / observable sockets)
df['rate'] = df['feature3'] / df['feature4']

# Binary indicator for human vs non-human primates
# Homo sapiens is human, others are non-human
df['is_human'] = (df['feature8'] == 'Homo sapiens').astype(int)

# Ensure categorical vars
# feature1 tooth class
# feature8 genus

# Model 1: binary human vs non-human, controlling for age, sex, tooth class
model1 = smf.ols('rate ~ is_human + feature5 + feature7 + C(feature1)', data=df).fit(cov_type='HC3')

# Model 2: genus categorical to compare Homo vs each genus
model2 = smf.ols('rate ~ C(feature8) + feature5 + feature7 + C(feature1)', data=df).fit(cov_type='HC3')

# Compute contrast: Homo sapiens vs average of non-human genera (Pan, Pongo, Papio)
# Using model2 with default reference (alphabetical) - we will build a contrast matrix
# Identify params
params = model2.params.index.tolist()

# Build contrast vector for Homo sapiens vs mean of other genera
# In patsy, C(feature8)[T.Homo sapiens], C(feature8)[T.Pan], C(feature8)[T.Papio], C(feature8)[T.Pongo]
# Reference category is alphabetical of feature8 levels; find which is reference
levels = sorted(df['feature8'].unique())
# The reference is the first level
ref = levels[0]

# Function to get mean for a level
# Mean = intercept + coef for level if not reference

def coef_for_level(level):
    if level == ref:
        return None
    return f"C(feature8)[T.{level}]"

levels_nonhuman = [lvl for lvl in levels if lvl != 'Homo sapiens']

# Build contrast: Homo - average(nonhuman)
contrast = np.zeros(len(params))

# intercept contributes to all levels equally; cancels in difference
# add Homo
homo_term = coef_for_level('Homo sapiens')
if homo_term is not None and homo_term in params:
    contrast[params.index(homo_term)] = 1.0

# subtract average of nonhuman
for lvl in levels_nonhuman:
    term = coef_for_level(lvl)
    if term is not None and term in params:
        contrast[params.index(term)] -= 1.0 / len(levels_nonhuman)
    else:
        # if lvl is reference, its coef is 0 in params; subtracting handled by intercept cancellation
        pass

contrast_test = model2.t_test(contrast)

# Print key results
print('Model1 (is_human) coef and p-value:')
print(model1.params['is_human'], model1.pvalues['is_human'])
print('Model1 95% CI:', model1.conf_int().loc['is_human'].tolist())
print('Model1 nobs:', int(model1.nobs))
print('\nModel2 Homo vs avg nonhuman contrast:')
print('contrast estimate:', float(contrast_test.effect))
print('contrast p-value:', float(contrast_test.pvalue))
print('contrast CI:', contrast_test.conf_int().tolist())

# Also compare Homo vs each genus directly
for lvl in levels_nonhuman:
    # difference Homo - lvl
    c = np.zeros(len(params))
    if homo_term is not None and homo_term in params:
        c[params.index(homo_term)] = 1.0
    term = coef_for_level(lvl)
    if term is not None and term in params:
        c[params.index(term)] -= 1.0
    # if lvl is reference, term not in params; subtracting ref means no change
    test = model2.t_test(c)
    print(f"Homo vs {lvl}: est {float(test.effect):.6f}, p {float(test.pvalue):.6f}")

