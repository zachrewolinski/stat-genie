import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import patsy
from scipy import stats

# Load data
_df = pd.read_csv('amtl.csv')

# Keep relevant columns
cols = ['num_amtl', 'age', 'prob_male', 'tooth_class', 'genus', 'specimen']
_df = _df[cols].dropna().copy()

# Fit OLS with categorical controls; cluster-robust SE by specimen
formula = 'num_amtl ~ C(genus) + age + prob_male + C(tooth_class)'
model = smf.ols(formula, data=_df).fit(cov_type='cluster', cov_kwds={'groups': _df['specimen']})

# Build design info for marginal means
info = model.model.data.design_info

# Compute mean design vector for each genus (marginalizing over observed covariates)

def mean_exog_for_genus(genus):
    df_g = _df.copy()
    df_g['genus'] = genus
    exog = patsy.build_design_matrices([info], df_g, return_type='dataframe')[0]
    xbar = exog.mean(axis=0).values
    return xbar, exog.columns

beta = model.params.values
cov = model.cov_params().values

genera = ['Homo sapiens', 'Pan', 'Papio', 'Pongo']
mean_preds = {}

for g in genera:
    xbar, cols_exog = mean_exog_for_genus(g)
    mean_preds[g] = float(xbar @ beta)

# Differences: Homo minus others
homo_xbar, cols_exog = mean_exog_for_genus('Homo sapiens')

results = []
for g in ['Pan', 'Papio', 'Pongo']:
    other_xbar, _ = mean_exog_for_genus(g)
    v = homo_xbar - other_xbar
    diff = float(v @ beta)
    se = float(np.sqrt(v @ cov @ v))
    t_stat = diff / se if se > 0 else np.nan
    pval = 2 * (1 - stats.t.cdf(abs(t_stat), df=model.df_resid)) if se > 0 else np.nan
    results.append((g, diff, se, t_stat, pval))

print('N:', len(_df))
print('Model R2:', model.rsquared)
print('Adjusted means by genus (marginalized):')
for g, v in mean_preds.items():
    print(g, v)

print('\nDifferences (Homo sapiens minus other genus):')
for g, diff, se, t, p in results:
    print(g, 'diff', diff, 'se', se, 't', t, 'p', p)
