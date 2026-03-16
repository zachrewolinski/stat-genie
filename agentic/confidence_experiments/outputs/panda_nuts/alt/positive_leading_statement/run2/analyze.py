import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_cluster
from scipy import stats


df = pd.read_csv('panda_nuts.csv')

# Efficiency: nuts opened per minute
# (per minute for interpretability; equivalent to per second up to a constant)
df['efficiency'] = df['nuts_opened'] / df['seconds'] * 60.0

# OLS with cluster-robust SE by chimpanzee (repeated sessions per individual)
ols_model = smf.ols('efficiency ~ age + C(sex) + C(help)', data=df).fit(
    cov_type='cluster', cov_kwds={'groups': df['chimpanzee']}
)

# Count model with time offset: nuts opened ~ predictors + log(seconds)
poisson_model = smf.glm(
    'nuts_opened ~ age + C(sex) + C(help)',
    data=df,
    family=sm.families.Poisson(),
    offset=np.log(df['seconds'])
).fit()

# Negative binomial as sensitivity (handles overdispersion)
nb_model = smf.glm(
    'nuts_opened ~ age + C(sex) + C(help)',
    data=df,
    family=sm.families.NegativeBinomial(alpha=1.0),
    offset=np.log(df['seconds'])
).fit()

# Cluster-robust covariance for GLM using sandwich estimator
poisson_cov = cov_cluster(poisson_model, df['chimpanzee'])
nb_cov = cov_cluster(nb_model, df['chimpanzee'])


def summarize_model(result, cov=None):
    params = result.params
    if cov is None:
        cov = result.cov_params()
    se = np.sqrt(np.diag(cov))
    z = params / se
    pvals = 2 * (1 - stats.norm.cdf(np.abs(z)))
    crit = stats.norm.ppf(0.975)
    conf_low = params - crit * se
    conf_high = params + crit * se
    out = {}
    for term in params.index:
        out[term] = {
            'coef': float(params[term]),
            'p': float(pvals[params.index.get_loc(term)]),
            'ci_low': float(conf_low[params.index.get_loc(term)]),
            'ci_high': float(conf_high[params.index.get_loc(term)]),
        }
    return out

# Overdispersion check for Poisson
pearson_chi2 = np.sum(((df['nuts_opened'] - poisson_model.fittedvalues) / np.sqrt(poisson_model.fittedvalues.clip(lower=1e-9))) ** 2)
poisson_dispersion = pearson_chi2 / poisson_model.df_resid

summary = {
    'n_rows': int(df.shape[0]),
    'n_chimps': int(df['chimpanzee'].nunique()),
    'efficiency_summary': df['efficiency'].describe().to_dict(),
    'ols_cluster': {
        term: {
            'coef': float(ols_model.params[term]),
            'p': float(ols_model.pvalues[term]),
            'ci_low': float(ols_model.conf_int().loc[term, 0]),
            'ci_high': float(ols_model.conf_int().loc[term, 1]),
        }
        for term in ols_model.params.index
    },
    'poisson_cluster': summarize_model(poisson_model, cov=poisson_cov),
    'poisson_dispersion': float(poisson_dispersion),
    'negbin_cluster': summarize_model(nb_model, cov=nb_cov),
}

with open('analysis_output.json', 'w') as f:
    json.dump(summary, f, indent=2)

print('Wrote analysis_output.json')
