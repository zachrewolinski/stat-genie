import json
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('panda_nuts.csv')

# Basic derived metric: efficiency (nuts opened per second)
_df['efficiency'] = _df['nuts_opened'] / _df['seconds']

# Summary stats
summary = _df[['nuts_opened', 'seconds', 'efficiency', 'age']].describe().to_dict()

# Group means for descriptive context
means_by_sex = _df.groupby('sex')['efficiency'].mean().to_dict()
means_by_help = _df.groupby('help')['efficiency'].mean().to_dict()

# Correlation (Spearman) between age and efficiency
spearman_r, spearman_p = stats.spearmanr(_df['age'], _df['efficiency'])

# OLS on efficiency with robust SEs
ols = smf.ols('efficiency ~ age + C(sex) + C(help)', data=_df).fit(cov_type='HC3')
ols_params = ols.params.to_dict()
ols_pvalues = ols.pvalues.to_dict()
ols_ci = ols.conf_int().rename(columns={0: 'low', 1: 'high'}).to_dict(orient='index')

# Poisson GLM on counts with offset log(seconds) to model rate
# Add small epsilon to avoid log(0) if any (shouldn't be)
offset = np.log(_df['seconds'].values)
pois = smf.glm('nuts_opened ~ age + C(sex) + C(help)', data=_df,
               family=sm.families.Poisson(), offset=offset).fit(cov_type='HC3')
pois_params = pois.params.to_dict()
pois_pvalues = pois.pvalues.to_dict()
pois_ci = pois.conf_int().rename(columns={0: 'low', 1: 'high'}).to_dict(orient='index')

# Convert Poisson coefficients to rate ratios
rate_ratios = {k: float(np.exp(v)) for k, v in pois_params.items()}
rr_ci = {k: (float(np.exp(v['low'])), float(np.exp(v['high']))) for k, v in pois_ci.items()}

# Save key outputs for inspection
result = {
    'n_rows': int(_df.shape[0]),
    'means_by_sex_efficiency': means_by_sex,
    'means_by_help_efficiency': means_by_help,
    'spearman_age_efficiency': {'rho': float(spearman_r), 'p': float(spearman_p)},
    'ols': {
        'params': ols_params,
        'pvalues': ols_pvalues,
        'ci': ols_ci,
        'r2': float(ols.rsquared),
        'adj_r2': float(ols.rsquared_adj),
    },
    'poisson': {
        'params': pois_params,
        'pvalues': pois_pvalues,
        'ci': pois_ci,
        'rate_ratios': rate_ratios,
        'rr_ci': rr_ci,
        'aic': float(pois.aic),
    },
    'summary': summary,
}

print(json.dumps(result, indent=2, sort_keys=True))
