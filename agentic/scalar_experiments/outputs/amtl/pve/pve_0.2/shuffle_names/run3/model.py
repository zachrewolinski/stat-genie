import pandas as pd
import statsmodels.formula.api as smf
import numpy as np

# Load data
_df = pd.read_csv('amtl.csv')

# Map columns based on observed values
# sockets: tooth class (Anterior/Posterior/Premolar)
# tooth_class: genus (Homo sapiens, Pan, Papio, Pongo)
# prob_male: specimen id
# pop: age estimate
# stdev_age: sex probability
# genus: AMTL measure (missing teeth count / frequency)

_df = _df.rename(columns={
    'sockets': 'tooth_class',
    'tooth_class': 'genus_cat',
    'prob_male': 'specimen_id',
    'pop': 'age_est',
    'stdev_age': 'sex_prob',
    'genus': 'amtl',
    'age': 'sockets_n',
    'num_amtl': 'age_uncert'
})

# Drop any rows with missing values (none expected)
_df = _df.dropna()

# OLS with cluster-robust SE by specimen
model = smf.ols('amtl ~ C(genus_cat) + age_est + sex_prob + C(tooth_class)', data=_df)
res = model.fit(cov_type='cluster', cov_kwds={'groups': _df['specimen_id']})
print(res.summary())

# Compute adjusted means per genus category at mean covariates
means = {}
for g in _df['genus_cat'].unique():
    temp = _df.copy()
    temp['genus_cat'] = g
    # predictions
    means[g] = res.predict(temp).mean()

print('\nAdjusted mean AMTL by genus:')
for k, v in sorted(means.items(), key=lambda x: x[1], reverse=True):
    print(k, v)

# Pairwise contrasts: Homo sapiens vs others
# Use statsmodels to get coefficient differences
params = res.params
cov = res.cov_params()

# baseline is first category in alphabetical order by default; but we can compute differences explicitly.
# We'll compute differences between Homo sapiens and each other genus using design matrix.
from patsy import dmatrix

# Build design row for each genus at mean covariates, reference tooth_class as most frequent
mean_age = _df['age_est'].mean()
mean_sex = _df['sex_prob'].mean()
# choose most frequent tooth_class
ref_tooth = _df['tooth_class'].value_counts().idxmax()

# Build a small dataframe for prediction
import itertools

genera = sorted(_df['genus_cat'].unique())
base_df = pd.DataFrame({
    'genus_cat': genera,
    'age_est': mean_age,
    'sex_prob': mean_sex,
    'tooth_class': ref_tooth
})
X = dmatrix(res.model.data.design_info.builder, base_df, return_type='dataframe')

# function to compute contrast

def contrast(i, j):
    diff = X.iloc[i] - X.iloc[j]
    est = float(diff @ params)
    se = float(np.sqrt(diff @ cov @ diff.T))
    z = est / se if se > 0 else np.nan
    # two-sided p-value
    from scipy import stats
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return est, se, z, p

print('\nContrasts (Homo sapiens minus other genera) at mean covariates:')
idx_map = {g:i for i,g in enumerate(genera)}
for other in genera:
    if other == 'Homo sapiens':
        continue
    est, se, z, p = contrast(idx_map['Homo sapiens'], idx_map[other])
    print(f'Homo sapiens - {other}: est={est:.3f}, se={se:.3f}, z={z:.3f}, p={p:.3g}')
