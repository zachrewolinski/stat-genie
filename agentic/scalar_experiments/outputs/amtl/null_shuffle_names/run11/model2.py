import pandas as pd
import numpy as np
import statsmodels.api as sm
from patsy import dmatrix, build_design_matrices
from scipy.stats import norm

# Load data
path = 'amtl.csv'
df = pd.read_csv(path)

# Map columns based on inference
missing = df['genus']
sockets = df['age']

# Remove invalid rows
valid = missing.between(0, sockets)
df = df[valid].copy()
missing = missing[valid]
sockets = sockets[valid]

# predictors
# genus category (Homo sapiens, Pan, Pongo, Papio)
df['genus_cat'] = df['tooth_class']
# tooth class (Anterior/Posterior/Premolar)
df['tooth_class_cat'] = df['sockets']
# age at death
df['age_years'] = df['pop']
# sex probability
_df_prob_male = df['stdev_age']
# ensure numeric
_df_prob_male = pd.to_numeric(_df_prob_male, errors='coerce')
df['prob_male'] = _df_prob_male

# Design matrix
formula = '1 + C(genus_cat) + age_years + prob_male + C(tooth_class_cat)'
X = dmatrix(formula, df, return_type='dataframe')
design_info = X.design_info

# Endog as counts
endog = np.column_stack([missing.values, (sockets - missing).values])

model = sm.GLM(endog, X, family=sm.families.Binomial())
res = model.fit()
print(res.summary())

# Identify baseline categories
cats = list(df['genus_cat'].astype('category').cat.categories)
print('Genus categories:', cats)

# Baseline category is first in alphabetical order by patsy
baseline = cats[0]

# parameter naming

def cat_param(cat):
    return f'C(genus_cat)[T.{cat}]'

# compute hs vs others
hs = 'Homo sapiens'
params = res.params
cov = res.cov_params()

# hs coef/var when not baseline
hs_coef = params.get(cat_param(hs), 0.0) if hs != baseline else 0.0
hs_var = cov.loc[cat_param(hs), cat_param(hs)] if hs != baseline else 0.0

results = []
for other in cats:
    if other == hs:
        continue
    if hs == baseline:
        # diff = baseline - other = -coef_other
        other_coef = params.get(cat_param(other), 0.0)
        diff = -other_coef
        var = cov.loc[cat_param(other), cat_param(other)]
    else:
        if other == baseline:
            diff = hs_coef
            var = hs_var
        else:
            other_coef = params.get(cat_param(other), 0.0)
            diff = hs_coef - other_coef
            var = hs_var + cov.loc[cat_param(other), cat_param(other)] - 2 * cov.loc[cat_param(hs), cat_param(other)]
    se = np.sqrt(var) if var >= 0 else np.nan
    z = diff / se if se and se > 0 else np.nan
    p = 2 * (1 - norm.cdf(abs(z))) if np.isfinite(z) else np.nan
    results.append((other, diff, se, z, p))

print('\nHomo sapiens vs others (log-odds diff):')
for other, diff, se, z, p in results:
    print(other, 'diff', diff, 'se', se, 'z', z, 'p', p)

# Predicted probability at mean covariates for baseline tooth class
mean_age = df['age_years'].mean()
mean_prob_male = df['prob_male'].mean()

# baseline tooth class
tooth_cats = list(df['tooth_class_cat'].astype('category').cat.categories)
base_tooth = tooth_cats[0]
print('Tooth classes:', tooth_cats, 'baseline', base_tooth)

# build function to create design row

def make_row(genus_cat, tooth_cat=base_tooth, age=mean_age, prob_male=mean_prob_male):
    tmp = pd.DataFrame({
        'genus_cat': [genus_cat],
        'tooth_class_cat': [tooth_cat],
        'age_years': [age],
        'prob_male': [prob_male],
    })
    row = build_design_matrices([design_info], tmp, return_type='dataframe')[0]
    return row

print('\nPredicted AMTL probability at mean age/sex, base tooth class:')
for cat in cats:
    row = make_row(cat)
    lp = float(np.dot(row.values, params))
    p = 1 / (1 + np.exp(-lp))
    print(cat, p)
