import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Load data
_df = pd.read_csv('panda_nuts.csv')

# Create efficiency metric: nuts opened per second
_df['efficiency'] = _df['nuts_opened'] / _df['seconds']

# Encode categorical variables
_df['sex'] = _df['sex'].astype('category')
_df['help'] = _df['help'].astype('category')
_df['hammer'] = _df['hammer'].astype('category')

# Basic summaries
summary = {
    'n_rows': int(_df.shape[0]),
    'efficiency_mean': float(_df['efficiency'].mean()),
    'efficiency_std': float(_df['efficiency'].std()),
    'efficiency_min': float(_df['efficiency'].min()),
    'efficiency_max': float(_df['efficiency'].max()),
    'sex_counts': _df['sex'].value_counts().to_dict(),
    'help_counts': _df['help'].value_counts().to_dict(),
    'efficiency_by_sex': _df.groupby('sex')['efficiency'].mean().to_dict(),
    'efficiency_by_help': _df.groupby('help')['efficiency'].mean().to_dict(),
    'age_efficiency_corr': float(_df['age'].corr(_df['efficiency'])),
}

# Regression: efficiency ~ age + sex + help (+ hammer as covariate)
model_basic = smf.ols('efficiency ~ age + C(sex) + C(help)', data=_df).fit(cov_type='HC3')
model_with_hammer = smf.ols('efficiency ~ age + C(sex) + C(help) + C(hammer)', data=_df).fit(cov_type='HC3')

# Count models for nuts opened with exposure (seconds)
_df['log_seconds'] = np.log(_df['seconds'])
glm_poisson = smf.glm(
    'nuts_opened ~ age + C(sex) + C(help)',
    data=_df,
    family=sm.families.Poisson(),
    offset=_df['log_seconds'],
).fit(cov_type='HC3')
glm_poisson_hammer = smf.glm(
    'nuts_opened ~ age + C(sex) + C(help) + C(hammer)',
    data=_df,
    family=sm.families.Poisson(),
    offset=_df['log_seconds'],
).fit(cov_type='HC3')

# Negative binomial to account for over-dispersion
glm_nb = smf.glm(
    'nuts_opened ~ age + C(sex) + C(help)',
    data=_df,
    family=sm.families.NegativeBinomial(),
    offset=_df['log_seconds'],
).fit(cov_type='HC3')
glm_nb_hammer = smf.glm(
    'nuts_opened ~ age + C(sex) + C(help) + C(hammer)',
    data=_df,
    family=sm.families.NegativeBinomial(),
    offset=_df['log_seconds'],
).fit(cov_type='HC3')

# Extract key stats
params_basic = model_basic.params
pvalues_basic = model_basic.pvalues
params_hammer = model_with_hammer.params
pvalues_hammer = model_with_hammer.pvalues

results = {
    'summary': summary,
    'basic': {
        'params': params_basic.to_dict(),
        'pvalues': pvalues_basic.to_dict(),
        'r2': float(model_basic.rsquared),
        'adj_r2': float(model_basic.rsquared_adj),
    },
    'with_hammer': {
        'params': params_hammer.to_dict(),
        'pvalues': pvalues_hammer.to_dict(),
        'r2': float(model_with_hammer.rsquared),
        'adj_r2': float(model_with_hammer.rsquared_adj),
    },
    'glm_poisson': {
        'params': glm_poisson.params.to_dict(),
        'pvalues': glm_poisson.pvalues.to_dict(),
        'aic': float(glm_poisson.aic),
    },
    'glm_poisson_hammer': {
        'params': glm_poisson_hammer.params.to_dict(),
        'pvalues': glm_poisson_hammer.pvalues.to_dict(),
        'aic': float(glm_poisson_hammer.aic),
    },
    'glm_nb': {
        'params': glm_nb.params.to_dict(),
        'pvalues': glm_nb.pvalues.to_dict(),
        'aic': float(glm_nb.aic),
    },
    'glm_nb_hammer': {
        'params': glm_nb_hammer.params.to_dict(),
        'pvalues': glm_nb_hammer.pvalues.to_dict(),
        'aic': float(glm_nb_hammer.aic),
    },
}

print(json.dumps(results, indent=2, sort_keys=True))
