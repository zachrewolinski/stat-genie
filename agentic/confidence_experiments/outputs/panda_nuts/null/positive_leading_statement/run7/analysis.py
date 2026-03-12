import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Load data
_df = pd.read_csv('panda_nuts.csv')

# Basic derived metrics
_df['rate_per_sec'] = _df['nuts_opened'] / _df['seconds']
_df['rate_per_min'] = _df['rate_per_sec'] * 60.0

# Ensure categorical types
_df['sex'] = _df['sex'].astype('category')
_df['help'] = _df['help'].astype('category')

# Poisson rate model with exposure (seconds)
poisson_model = smf.glm(
    'nuts_opened ~ age + C(sex) + C(help)',
    data=_df,
    family=sm.families.Poisson(),
    offset=np.log(_df['seconds'])
)
poisson_res = poisson_model.fit(cov_type='HC3')

# Overdispersion check
pearson_chi2 = poisson_res.pearson_chi2
od_ratio = pearson_chi2 / poisson_res.df_resid

# Negative binomial (if needed for robustness)
nb_model = smf.glm(
    'nuts_opened ~ age + C(sex) + C(help)',
    data=_df,
    family=sm.families.NegativeBinomial(alpha=1.0),
    offset=np.log(_df['seconds'])
)
nb_res = nb_model.fit(cov_type='HC3')

# Linear model on rate per minute (for interpretability)
ols_model = smf.ols('rate_per_min ~ age + C(sex) + C(help)', data=_df)
ols_res = ols_model.fit(cov_type='HC3')

# Summaries
results = {
    'poisson': {
        'params': poisson_res.params,
        'pvalues': poisson_res.pvalues,
        'conf_int': poisson_res.conf_int(),
        'od_ratio': od_ratio
    },
    'negbin': {
        'params': nb_res.params,
        'pvalues': nb_res.pvalues,
        'conf_int': nb_res.conf_int(),
    },
    'ols': {
        'params': ols_res.params,
        'pvalues': ols_res.pvalues,
        'conf_int': ols_res.conf_int(),
        'r2': ols_res.rsquared
    }
}

# Print compact summaries for review
print('Rows:', len(_df))
print('Sex counts:\n', _df['sex'].value_counts())
print('Help counts:\n', _df['help'].value_counts())
print('\nPoisson HC3 p-values:\n', results['poisson']['pvalues'])
print('Poisson overdispersion ratio:', round(od_ratio, 3))
print('\nNegBin HC3 p-values:\n', results['negbin']['pvalues'])
print('\nOLS HC3 p-values:\n', results['ols']['pvalues'])

# Effect sizes as rate ratios for GLM
rate_ratios = np.exp(poisson_res.params)
rr_ci = np.exp(results['poisson']['conf_int'])
print('\nPoisson rate ratios (RR) and 95% CI:')
for k in rate_ratios.index:
    lo, hi = rr_ci.loc[k]
    print(f'{k}: RR={rate_ratios[k]:.3f} (CI {lo:.3f}, {hi:.3f})')

