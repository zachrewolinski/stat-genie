import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm

# load

df = pd.read_csv('amtl.csv')

# build binary human indicator

df['is_human'] = (df['genus'] == 'Homo sapiens').astype(int)

# OLS with human indicator
model_bin = smf.ols('num_amtl ~ is_human + age + prob_male + C(tooth_class)', data=df).fit(cov_type='HC3')
print(model_bin.summary())

# OLS with genus categories, set baseline Papio
model_genus = smf.ols('num_amtl ~ C(genus, Treatment(reference="Papio")) + age + prob_male + C(tooth_class)', data=df).fit(cov_type='HC3')
print(model_genus.summary())

# overall ANOVA for genus
anova = anova_lm(model_genus, typ=2)
print(anova)

# Pairwise contrasts Homo vs each nonhuman
# using model_genus params: intercept is Papio baseline
params = model_genus.params
cov = model_genus.cov_params()

# Homo vs Papio is coefficient C(genus)[T.Homo sapiens]
# Homo vs Pan = (Homo coef - Pan coef)
# Homo vs Pongo = (Homo coef - Pongo coef)

import numpy as np

def contrast(name, coef_vec):
    est = float(coef_vec @ params)
    var = float(coef_vec @ cov @ coef_vec)
    se = var ** 0.5
    t = est / se
    df_resid = model_genus.df_resid
    p = 2 * (1 - sm.stats.t.cdf(abs(t), df_resid))
    return name, est, se, t, p

# build coefficient vector aligned to params index
index = params.index

def vec_for_diff(coef_name_a, coef_name_b=None):
    v = np.zeros(len(index))
    if coef_name_a:
        v[index.get_loc(coef_name_a)] = 1.0
    if coef_name_b:
        v[index.get_loc(coef_name_b)] -= 1.0
    return v

homo = 'C(genus, Treatment(reference="Papio"))[T.Homo sapiens]'
pan = 'C(genus, Treatment(reference="Papio"))[T.Pan]'
pongo = 'C(genus, Treatment(reference="Papio"))[T.Pongo]'

# Homo vs Papio
res1 = contrast('Homo vs Papio', vec_for_diff(homo))
# Homo vs Pan
res2 = contrast('Homo vs Pan', vec_for_diff(homo, pan))
# Homo vs Pongo
res3 = contrast('Homo vs Pongo', vec_for_diff(homo, pongo))

print('Contrasts:')
for r in [res1,res2,res3]:
    print(r)
