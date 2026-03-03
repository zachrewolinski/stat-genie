import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats


df = pd.read_csv('amtl.csv')
df['is_human'] = (df['genus'] == 'Homo sapiens').astype(int)

model_bin = smf.ols('num_amtl ~ is_human + age + prob_male + C(tooth_class)', data=df).fit(cov_type='HC3')

model_genus = smf.ols('num_amtl ~ C(genus, Treatment(reference="Papio")) + age + prob_male + C(tooth_class)', data=df).fit(cov_type='HC3')

# contrasts
params = model_genus.params
cov = model_genus.cov_params()
index = params.index


def contrast(coef_vec):
    est = float(coef_vec @ params)
    var = float(coef_vec @ cov @ coef_vec)
    se = var ** 0.5
    t = est / se
    df_resid = model_genus.df_resid
    p = 2 * (1 - stats.t.cdf(abs(t), df_resid))
    ci_low, ci_high = stats.t.interval(0.95, df_resid, loc=est, scale=se)
    return est, se, t, p, ci_low, ci_high

homo = 'C(genus, Treatment(reference="Papio"))[T.Homo sapiens]'
pan = 'C(genus, Treatment(reference="Papio"))[T.Pan]'
pongo = 'C(genus, Treatment(reference="Papio"))[T.Pongo]'


def vec_for_diff(coef_name_a, coef_name_b=None):
    v = np.zeros(len(index))
    if coef_name_a:
        v[index.get_loc(coef_name_a)] = 1.0
    if coef_name_b:
        v[index.get_loc(coef_name_b)] -= 1.0
    return v

contrasts = {
    'Homo vs Papio': vec_for_diff(homo),
    'Homo vs Pan': vec_for_diff(homo, pan),
    'Homo vs Pongo': vec_for_diff(homo, pongo),
}

print('Human indicator model:')
print(model_bin.summary())

print('\nGenus model:')
print(model_genus.summary())

print('\nContrasts (Homo vs nonhuman):')
for name, v in contrasts.items():
    est, se, t, p, ci_low, ci_high = contrast(v)
    print(name, 'est', est, 'se', se, 't', t, 'p', p, '95% CI', (ci_low, ci_high))

# effect size: standardized difference for human indicator
sd = df['num_amtl'].std()
coef = model_bin.params['is_human']
print('\nStandardized effect (human vs nonhuman):', coef / sd)
