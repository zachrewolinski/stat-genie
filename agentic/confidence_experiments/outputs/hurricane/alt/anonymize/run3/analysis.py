import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
import statsmodels.discrete.discrete_model as smd

# Load data
_df = pd.read_csv('hurricane.csv')

# Basic checks
missing = _df.isna().sum()

# Define variables
_df['log_deaths'] = np.log1p(_df['feature8'])

# Main femininity measures
_df['fem_index'] = _df['feature4']
_df['fem_index_mt'] = _df['feature12']
_df['female_binary'] = _df['feature6']

# Controls (storm intensity and year)
_df['category'] = _df['feature7']
_df['pressure'] = _df['feature5']
_df['wind'] = _df['feature13']
_df['year'] = _df['feature2']

# Helper to fit OLS with robust SE

def fit_ols(formula: str):
    model = smf.ols(formula, data=_df).fit(cov_type='HC3')
    return model

models = {}

# Model with femininity index (coder ratings)
models['fem_index'] = fit_ols('log_deaths ~ fem_index + category + pressure + wind + year')

# Model with binary female indicator
models['female_binary'] = fit_ols('log_deaths ~ female_binary + category + pressure + wind + year')

# Model with MTurk femininity ratings
models['fem_index_mt'] = fit_ols('log_deaths ~ fem_index_mt + category + pressure + wind + year')

# Simple bivariate correlations
corr_fem = _df['fem_index'].corr(_df['feature8'])
corr_fem_log = _df['fem_index'].corr(_df['log_deaths'])
corr_female_bin = _df['female_binary'].corr(_df['feature8'])

# GLM Poisson (as count-like outcome)
# Add small constant to avoid issues; Poisson can handle zero
poisson_model = smf.glm('feature8 ~ fem_index + category + pressure + wind + year', data=_df,
                        family=sm.families.Poisson()).fit(cov_type='HC3')

# Negative binomial to address over-dispersion
nb_model = smf.glm('feature8 ~ fem_index + category + pressure + wind + year', data=_df,
                   family=sm.families.NegativeBinomial()).fit(cov_type='HC3')

# Discrete negative binomial with estimated dispersion
exog = sm.add_constant(_df[['fem_index', 'category', 'pressure', 'wind', 'year']])
nb2_model = smd.NegativeBinomial(_df['feature8'], exog).fit(disp=False)

# Collect key stats
summary = {
    'n': int(len(_df)),
    'missing_total': int(missing.sum()),
    'corr_fem_deaths': float(corr_fem),
    'corr_fem_log_deaths': float(corr_fem_log),
    'corr_female_bin_deaths': float(corr_female_bin),
}

results = {}
for name, model in models.items():
    coef = model.params.get(name, np.nan)
    pval = model.pvalues.get(name, np.nan)
    results[name] = {
        'coef': float(coef),
        'pval': float(pval),
        'r2': float(model.rsquared),
    }

# Poisson coefficient/p-value
results['poisson_fem_index'] = {
    'coef': float(poisson_model.params.get('fem_index', np.nan)),
    'pval': float(poisson_model.pvalues.get('fem_index', np.nan)),
    'deviance_over_df': float(poisson_model.deviance / poisson_model.df_resid),
    'pseudo_r2_cs': float(poisson_model.prsquared) if hasattr(poisson_model, 'prsquared') else None,
}

results['neg_binom_fem_index'] = {
    'coef': float(nb_model.params.get('fem_index', np.nan)),
    'pval': float(nb_model.pvalues.get('fem_index', np.nan)),
    'deviance_over_df': float(nb_model.deviance / nb_model.df_resid),
}

results['neg_binom2_fem_index'] = {
    'coef': float(nb2_model.params.get('fem_index', np.nan)),
    'pval': float(nb2_model.pvalues.get('fem_index', np.nan)),
    'alpha': float(nb2_model.params.get('alpha', np.nan)),
}

# Save to json for inspection
import json
with open('analysis_results.json', 'w') as f:
    json.dump({'summary': summary, 'results': results}, f, indent=2)

print(json.dumps({'summary': summary, 'results': results}, indent=2))
