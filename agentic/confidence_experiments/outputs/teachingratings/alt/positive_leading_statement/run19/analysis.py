import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import scipy.stats as stats

# Load data
_df = pd.read_csv('teachingratings.csv')

# Basic correlation
pearson_r, pearson_p = stats.pearsonr(_df['beauty'], _df['eval'])

# Simple OLS
m_simple = smf.ols('eval ~ beauty', data=_df).fit(cov_type='HC3')

# Multivariate OLS with controls
# Encode categorical variables with C()
formula = (
    'eval ~ beauty + age + C(gender) + C(minority) + C(credits) + '
    'C(division) + C(native) + C(tenure) + students + allstudents'
)

m_controls_hc3 = smf.ols(formula, data=_df).fit(cov_type='HC3')

# Cluster-robust SEs by professor (repeated courses per instructor)
m_controls_cluster = smf.ols(formula, data=_df).fit(cov_type='cluster', cov_kwds={'groups': _df['prof']})

# Effect size: change in eval for 1 SD increase in beauty
beauty_sd = _df['beauty'].std(ddof=1)
coef_simple = m_simple.params['beauty']
coef_controls = m_controls_hc3.params['beauty']
coef_controls_cluster = m_controls_cluster.params['beauty']

sd_effect_simple = coef_simple * beauty_sd
sd_effect_controls = coef_controls * beauty_sd
sd_effect_controls_cluster = coef_controls_cluster * beauty_sd

results = {
    'pearson_r': pearson_r,
    'pearson_p': pearson_p,
    'simple_coef': coef_simple,
    'simple_p_hc3': m_simple.pvalues['beauty'],
    'controls_coef_hc3': coef_controls,
    'controls_p_hc3': m_controls_hc3.pvalues['beauty'],
    'controls_coef_cluster': coef_controls_cluster,
    'controls_p_cluster': m_controls_cluster.pvalues['beauty'],
    'sd_effect_simple': sd_effect_simple,
    'sd_effect_controls': sd_effect_controls,
    'sd_effect_controls_cluster': sd_effect_controls_cluster,
    'n': int(_df.shape[0]),
    'unique_prof': int(_df['prof'].nunique())
}

print(json.dumps(results, indent=2))
