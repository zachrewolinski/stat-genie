import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.stats as st
import patsy

# Load data
_df = pd.read_csv('amtl.csv')

# Keep relevant columns and drop missing
cols = ['num_amtl', 'sockets', 'age', 'prob_male', 'tooth_class', 'genus']
_df = _df[cols].dropna()

# Ensure categories
_df['tooth_class'] = _df['tooth_class'].astype('category')
_df['genus'] = _df['genus'].astype('category')

# Response as proportion with weights
_df['amtl_rate'] = _df['num_amtl'] / _df['sockets']

# Fit binomial GLM
formula = 'amtl_rate ~ C(genus) + age + prob_male + C(tooth_class)'
model = smf.glm(formula=formula, data=_df, family=sm.families.Binomial(), freq_weights=_df['sockets'])
res = model.fit()

# Marginal standardized predictions per genus
means = {}
for g in _df['genus'].cat.categories:
    tmp = _df.copy()
    tmp['genus'] = g
    pred = res.predict(tmp)
    means[g] = float(np.average(pred))

# Pairwise contrasts using design matrices at typical covariate values
mean_age = float(_df['age'].mean())
mean_prob_male = float(_df['prob_male'].mean())
mode_tooth = _df['tooth_class'].value_counts().idxmax()

base = pd.DataFrame({
    'age': [mean_age],
    'prob_male': [mean_prob_male],
    'tooth_class': [mode_tooth],
    'genus': [None],
})

design_info = res.model.data.design_info

def design_row(genus):
    row = base.copy()
    row['genus'] = genus
    exog = patsy.build_design_matrices([design_info], row)[0]
    return np.asarray(exog)[0]

cov = res.cov_params().values
params = res.params.values

def contrast(x1, x0):
    v = x1 - x0
    est = float(np.dot(v, params))
    se = float(np.sqrt(np.dot(v, np.dot(cov, v))))
    z = est / se if se > 0 else np.nan
    p = 2 * (1 - st.norm.cdf(abs(z))) if se > 0 else np.nan
    return est, se, z, p

results = {}
for g in _df['genus'].cat.categories:
    if g == 'Homo sapiens':
        continue
    x_h = design_row('Homo sapiens')
    x_g = design_row(g)
    est, se, z, p = contrast(x_h, x_g)
    results[g] = (est, se, z, p)

print('N rows:', len(_df))
print('Genus counts:', _df['genus'].value_counts().to_dict())
print('Adjusted mean AMTL rate by genus:')
for g, m in means.items():
    print(f'  {g}: {m:.4f}')
print('\nHomo sapiens vs each nonhuman genus (log-odds difference at typical covariates):')
for g, (est, se, z, p) in results.items():
    print(f'  Homo vs {g}: est={est:.3f}, se={se:.3f}, z={z:.2f}, p={p:.4g}')

nonhuman = [g for g in means if g != 'Homo sapiens']
nonhuman_mean = float(np.mean([means[g] for g in nonhuman]))
print(f"\nNonhuman mean adjusted rate (unweighted avg): {nonhuman_mean:.4f}")
print(f"Homo adjusted rate: {means.get('Homo sapiens', float('nan')):.4f}")
