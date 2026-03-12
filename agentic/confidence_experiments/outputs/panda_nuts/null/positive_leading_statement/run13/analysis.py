import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('panda_nuts.csv')

# Basic cleaning
_df['sex'] = _df['sex'].astype('category')
_df['help'] = _df['help'].astype('category')

# Efficiency as nuts per second
_df['rate'] = _df['nuts_opened'] / _df['seconds']

# Poisson GLM with offset for exposure time
# This models nuts_opened as a rate per second
poisson_model = smf.glm(
    formula='nuts_opened ~ age + sex + help',
    data=_df,
    family=sm.families.Poisson(),
    offset=np.log(_df['seconds'])
).fit(cov_type='HC0')

# Negative Binomial (NB2) with offset to address overdispersion
nb_model = smf.negativebinomial(
    formula='nuts_opened ~ age + sex + help',
    data=_df,
    exposure=_df['seconds']
).fit(disp=False)

# Linear model on rate for a simple sanity check
ols_model = smf.ols('rate ~ age + sex + help', data=_df).fit(cov_type='HC3')

# Extract key results
poisson_params = poisson_model.params
poisson_pvalues = poisson_model.pvalues
poisson_ci = poisson_model.conf_int()

results = {
    'poisson': {
        'params': poisson_params.to_dict(),
        'pvalues': poisson_pvalues.to_dict(),
        'ci': {k: [float(v[0]), float(v[1])] for k, v in poisson_ci.iterrows()},
        'n': int(_df.shape[0]),
    },
    'negative_binomial': {
        'params': nb_model.params.to_dict(),
        'pvalues': nb_model.pvalues.to_dict(),
        'n': int(_df.shape[0]),
    },
    'ols': {
        'params': ols_model.params.to_dict(),
        'pvalues': ols_model.pvalues.to_dict(),
        'r2': float(ols_model.rsquared),
    },
    'help_counts': _df['help'].value_counts().to_dict(),
    'sex_counts': _df['sex'].value_counts().to_dict(),
}

with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(poisson_model.summary())
print('\nNegative Binomial summary:')
print(nb_model.summary())
print('\nOLS summary (rate):')
print(ols_model.summary())
