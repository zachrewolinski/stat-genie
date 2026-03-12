import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

# Load data
csv_path = 'panda_nuts.csv'
df = pd.read_csv(csv_path)

# Clean
# Ensure categorical types
for col in ['sex', 'help', 'hammer']:
    if col in df.columns:
        df[col] = df[col].astype('category')

# Define efficiency (nuts per second)
df['efficiency'] = df['nuts_opened'] / df['seconds']

# Basic summary
summary = {
    'n_rows': len(df),
    'missing': df.isna().sum().to_dict(),
    'efficiency_mean': df['efficiency'].mean(),
    'efficiency_std': df['efficiency'].std(),
}

# Poisson regression for rate with offset log(seconds)
# This models expected nuts_opened ~ exp(Xb) * seconds
# Use robust SE to mitigate mild misspecification
formula = 'nuts_opened ~ age + C(sex) + C(help)'
model_pois = smf.glm(
    formula=formula,
    data=df,
    family=sm.families.Poisson(),
    offset=np.log(df['seconds'])
).fit(cov_type='HC3')

# Overdispersion check
pearson_chi2 = model_pois.pearson_chi2
pearson_df = model_pois.df_resid
overdispersion = pearson_chi2 / pearson_df if pearson_df > 0 else np.nan

# Negative binomial as sensitivity, estimate alpha via discrete model
# Use statsmodels NegativeBinomial (NB2) with log link
try:
    import statsmodels.discrete.discrete_model as smd
    y = df['nuts_opened']
    X = sm.add_constant(pd.get_dummies(df[['age', 'sex', 'help']], drop_first=True))
    # offset log(seconds)
    nb_model = smd.NegativeBinomial(y, X, loglike_method='nb2', offset=np.log(df['seconds']))
    nb_res = nb_model.fit(disp=False)
except Exception as e:
    nb_res = None
    nb_error = str(e)
else:
    nb_error = None

# Also OLS on efficiency for interpretability
ols_model = smf.ols('efficiency ~ age + C(sex) + C(help)', data=df).fit(cov_type='HC3')

# Extract results

def summarize_glm(res):
    params = res.params
    bse = res.bse
    pvals = res.pvalues
    # Convert to rate ratios
    rate_ratio = np.exp(params)
    out = {}
    for k in params.index:
        out[k] = {
            'coef': float(params[k]),
            'se': float(bse[k]),
            'p': float(pvals[k]),
            'rate_ratio': float(rate_ratio[k])
        }
    return out


def summarize_ols(res):
    params = res.params
    bse = res.bse
    pvals = res.pvalues
    out = {}
    for k in params.index:
        out[k] = {
            'coef': float(params[k]),
            'se': float(bse[k]),
            'p': float(pvals[k])
        }
    return out

results = {
    'summary': summary,
    'poisson': {
        'overdispersion_ratio': float(overdispersion),
        'params': summarize_glm(model_pois),
        'aic': float(model_pois.aic),
    },
    'ols': {
        'params': summarize_ols(ols_model),
        'r2': float(ols_model.rsquared),
        'r2_adj': float(ols_model.rsquared_adj),
    },
}

if nb_res is not None:
    params = nb_res.params
    bse = nb_res.bse
    pvals = nb_res.pvalues
    rate_ratio = np.exp(params)
    nb_params = {}
    for k in params.index:
        nb_params[k] = {
            'coef': float(params[k]),
            'se': float(bse[k]),
            'p': float(pvals[k]),
            'rate_ratio': float(rate_ratio[k])
        }
    results['neg_bin'] = {
        'params': nb_params,
        'aic': float(nb_res.aic),
        'alpha': float(nb_res.params.get('alpha', np.nan))
    }
else:
    results['neg_bin_error'] = nb_error

# Multiple testing adjustment (3 predictors) on Poisson p-values excluding intercept
predictor_p = []
labels = []
for k, v in results['poisson']['params'].items():
    if k == 'Intercept':
        continue
    labels.append(k)
    predictor_p.append(v['p'])

if predictor_p:
    adj = multipletests(predictor_p, method='fdr_bh')
    results['poisson']['pvals_fdr_bh'] = {lab: float(p) for lab, p in zip(labels, adj[1])}

with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
