import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('panda_nuts.csv')

# Basic cleaning
# ensure categorical types
_df['sex'] = _df['sex'].astype('category')
_df['help'] = _df['help'].astype('category')

# Efficiency = nuts opened per second
_df['efficiency'] = _df['nuts_opened'] / _df['seconds']

# Poisson regression for counts with exposure (seconds)
# Using GLM with log(seconds) as offset to model rate
# Also run OLS on efficiency for interpretability

# GLM Poisson
_df['log_seconds'] = np.log(_df['seconds'])

poisson_model = smf.glm(
    formula='nuts_opened ~ age + C(sex) + C(help)',
    data=_df,
    family=sm.families.Poisson(),
    offset=_df['log_seconds']
).fit()

# Overdispersion check (Pearson chi2 / df)
pearson_chi2 = poisson_model.pearson_chi2
pearson_dispersion = pearson_chi2 / poisson_model.df_resid

# If overdispersed, fit Negative Binomial as robustness
nb_model = smf.glm(
    formula='nuts_opened ~ age + C(sex) + C(help)',
    data=_df,
    family=sm.families.NegativeBinomial(alpha=1.0),
    offset=_df['log_seconds']
).fit()

# OLS on efficiency
ols_model = smf.ols('efficiency ~ age + C(sex) + C(help)', data=_df).fit()

# Collect key results
results = {
    'n': int(_df.shape[0]),
    'poisson': {
        'params': poisson_model.params.to_dict(),
        'pvalues': poisson_model.pvalues.to_dict(),
        'dispersion': float(pearson_dispersion),
    },
    'neg_bin': {
        'params': nb_model.params.to_dict(),
        'pvalues': nb_model.pvalues.to_dict(),
    },
    'ols': {
        'params': ols_model.params.to_dict(),
        'pvalues': ols_model.pvalues.to_dict(),
        'r2': float(ols_model.rsquared),
    },
    'efficiency_summary': {
        'mean': float(_df['efficiency'].mean()),
        'std': float(_df['efficiency'].std()),
        'min': float(_df['efficiency'].min()),
        'max': float(_df['efficiency'].max()),
    }
}

with open('analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
