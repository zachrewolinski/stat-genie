import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

path = 'amtl.csv'
df = pd.read_csv(path)

# Map columns to conceptual variables based on observed values
# genus category is in 'tooth_class'
# tooth class is in 'sockets'
# age at death in 'pop'
# sex probability in 'stdev_age'
# response (AMTL measure) in 'genus'

# create binary indicator for Homo sapiens

df['is_human'] = (df['tooth_class'] == 'Homo sapiens').astype(int)

# OLS with categorical genus (taxon) + controls
model_cat = smf.ols('genus ~ C(tooth_class) + pop + stdev_age + C(sockets)', data=df).fit(
    cov_type='cluster', cov_kwds={'groups': df['prob_male']}
)

# OLS with human vs non-human
model_bin = smf.ols('genus ~ is_human + pop + stdev_age + C(sockets)', data=df).fit(
    cov_type='cluster', cov_kwds={'groups': df['prob_male']}
)

print('Model with categorical genus (taxon):')
print(model_cat.summary().tables[1])

print('\nModel with human vs non-human:')
print(model_bin.summary().tables[1])

# Compute contrast: Homo sapiens vs mean of non-human genera
# using model_cat params
params = model_cat.params
# baseline is first category alphabetically? statsmodels uses alphabetical ordering for C
# Determine ordering
cats = sorted(df['tooth_class'].unique())
print('\nCategory order:', cats)

# Build contrast vector for Homo vs mean of others
# params include Intercept, C(tooth_class)[T.<cat>] for non-baseline categories
# compute predicted mean for each genus at average covariates (which cancel in difference)

baseline = cats[0]

# function to get effect for each genus relative to baseline

def genus_effect(genus):
    if genus == baseline:
        return 0.0
    return params.get(f'C(tooth_class)[T.{genus}]', 0.0)

# effects
homo = 'Homo sapiens'
non_humans = [g for g in cats if g != homo]

homo_eff = genus_effect(homo)
non_eff = np.mean([genus_effect(g) for g in non_humans])
contrast = homo_eff - non_eff

# approximate SE for contrast using covariance matrix
cov = model_cat.cov_params()

# build contrast vector
param_names = model_cat.params.index.tolist()
L = np.zeros(len(param_names))

# effect for Homo
if homo != baseline:
    L[param_names.index(f'C(tooth_class)[T.{homo}]')] += 1.0
# subtract mean of non-human effects
for g in non_humans:
    if g == baseline:
        # baseline effect is 0
        L += 0
    else:
        L[param_names.index(f'C(tooth_class)[T.{g}]')] -= 1.0 / len(non_humans)

# compute t-stat
contrast_se = float(np.sqrt(L @ cov.values @ L))
contrast_t = contrast / contrast_se
from scipy import stats
p_value = 2 * (1 - stats.t.cdf(abs(contrast_t), df=model_cat.df_resid))

print('\nHomo vs mean of non-human genera contrast:')
print('contrast', contrast)
print('SE', contrast_se)
print('t', contrast_t)
print('p', p_value)

# also report adjusted predicted means by genus at average covariates
means = {}
for g in cats:
    means[g] = model_cat.predict({
        'tooth_class': [g],
        'pop': [df['pop'].mean()],
        'stdev_age': [df['stdev_age'].mean()],
        'sockets': [df['sockets'].mode()[0]],
    })[0]

print('\nAdjusted predicted mean genus (AMTL measure) by genus category:')
for g, v in means.items():
    print(g, v)

