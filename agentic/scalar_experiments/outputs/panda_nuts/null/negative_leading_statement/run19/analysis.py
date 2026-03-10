import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_cluster
from scipy import stats

# Load data
csv_path = 'panda_nuts.csv'
df = pd.read_csv(csv_path)

# Efficiency as rate of nuts opened per second
# Avoid division by zero (none expected)
df['efficiency'] = df['nuts_opened'] / df['seconds']

# Basic summaries
summary = {
    'n_rows': int(df.shape[0]),
    'n_chimpanzees': int(df['chimpanzee'].nunique()),
    'efficiency_mean': float(df['efficiency'].mean()),
    'efficiency_std': float(df['efficiency'].std()),
}

# OLS on efficiency with cluster-robust SE by chimpanzee
formula_base = 'efficiency ~ age + C(sex) + C(help)'
model_base = smf.ols(formula_base, data=df).fit(
    cov_type='cluster', cov_kwds={'groups': df['chimpanzee']}
)

formula_ctrl = 'efficiency ~ age + C(sex) + C(help) + C(hammer)'
model_ctrl = smf.ols(formula_ctrl, data=df).fit(
    cov_type='cluster', cov_kwds={'groups': df['chimpanzee']}
)

# Poisson GLM with offset log(seconds) (rate model)
# Use cluster-robust covariance by chimpanzee for inference
poisson_base = smf.glm(
    'nuts_opened ~ age + C(sex) + C(help)',
    data=df,
    family=sm.families.Poisson(),
    offset=np.log(df['seconds'])
).fit()
# Cluster-robust covariance for GLM
poisson_base_cov = cov_cluster(poisson_base, df['chimpanzee'])

poisson_ctrl = smf.glm(
    'nuts_opened ~ age + C(sex) + C(help) + C(hammer)',
    data=df,
    family=sm.families.Poisson(),
    offset=np.log(df['seconds'])
).fit()
poisson_ctrl_cov = cov_cluster(poisson_ctrl, df['chimpanzee'])

# Helper to collect coefficients and p-values

def collect_results(res, label, cov=None):
    out = {}
    if cov is not None:
        # Compute robust SEs and p-values using normal approximation
        se = pd.Series(np.sqrt(np.diag(cov)), index=res.params.index)
        z = res.params / se
        pvals = pd.Series(2 * (1 - stats.norm.cdf(np.abs(z))), index=res.params.index)
    else:
        pvals = res.pvalues
    for term in ['age', 'C(sex)[T.m]', 'C(help)[T.y]']:
        if term in res.params.index:
            out[term] = {
                'coef': float(res.params[term]),
                'pvalue': float(pvals[term]),
            }
    return out

results = {
    'summary': summary,
    'ols_base': collect_results(model_base, 'ols_base'),
    'ols_ctrl': collect_results(model_ctrl, 'ols_ctrl'),
    'poisson_base': collect_results(poisson_base, 'poisson_base', cov=poisson_base_cov),
    'poisson_ctrl': collect_results(poisson_ctrl, 'poisson_ctrl', cov=poisson_ctrl_cov),
}

# Also include model fit diagnostics for context
results['ols_base_adj_r2'] = float(model_base.rsquared_adj)
results['ols_ctrl_adj_r2'] = float(model_ctrl.rsquared_adj)
results['poisson_base_aic'] = float(poisson_base.aic)
results['poisson_ctrl_aic'] = float(poisson_ctrl.aic)
results['poisson_base_dispersion'] = float(poisson_base.pearson_chi2 / poisson_base.df_resid)
results['poisson_ctrl_dispersion'] = float(poisson_ctrl.pearson_chi2 / poisson_ctrl.df_resid)

# Negative binomial (if available) to address overdispersion
nb_results = {}
nb_alpha = None
try:
    if hasattr(smf, 'negativebinomial'):
        nb_model = smf.negativebinomial(
            'nuts_opened ~ age + C(sex) + C(help) + C(hammer)',
            data=df,
            offset=np.log(df['seconds'])
        ).fit(disp=False)
        nb_cov = cov_cluster(nb_model, df['chimpanzee'])
        nb_results = collect_results(nb_model, 'negative_binomial', cov=nb_cov)
        if 'alpha' in nb_model.params.index:
            nb_alpha = float(nb_model.params['alpha'])
    else:
        nb_results = {'error': 'negativebinomial model not available in statsmodels.formula.api'}
except Exception as exc:
    nb_results = {'error': str(exc)}

results['negative_binomial'] = nb_results
if nb_alpha is not None:
    results['negative_binomial_alpha'] = nb_alpha

with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
