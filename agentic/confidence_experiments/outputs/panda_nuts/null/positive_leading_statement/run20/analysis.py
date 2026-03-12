import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('panda_nuts.csv')

# Compute efficiency (nuts per second)
_df['efficiency'] = _df['nuts_opened'] / _df['seconds']

# Ensure categories
_df['sex'] = _df['sex'].astype('category')
_df['help'] = _df['help'].astype('category')
_df['hammer'] = _df['hammer'].astype('category')

# OLS on efficiency with robust SEs
ols_formula = 'efficiency ~ age + sex + help + hammer'
ols_model = smf.ols(ols_formula, data=_df).fit(cov_type='HC3')

# Poisson GLM on counts with offset log(seconds)
_df['log_seconds'] = np.log(_df['seconds'])
poisson_formula = 'nuts_opened ~ age + sex + help + hammer'
poisson_model = smf.glm(
    poisson_formula,
    data=_df,
    family=sm.families.Poisson(),
    offset=_df['log_seconds']
).fit()

# Overdispersion check
pearson_chi2 = sum(poisson_model.resid_pearson**2)
pearson_df = poisson_model.df_resid
pearson_ratio = pearson_chi2 / pearson_df if pearson_df > 0 else np.nan

# Negative Binomial GLM if overdispersion suggests it
nb_model = None
if pearson_ratio > 1.5:
    nb_model = smf.glm(
        poisson_formula,
        data=_df,
        family=sm.families.NegativeBinomial(alpha=1.0),
        offset=_df['log_seconds']
    ).fit()

# Summarize key coefficients for age, sex, help

def coef_table(model, label):
    params = model.params
    conf = model.conf_int()
    pvals = model.pvalues
    out = pd.DataFrame({
        'coef': params,
        'p_value': pvals,
        'ci_low': conf[0],
        'ci_high': conf[1],
    })
    out['model'] = label
    return out

ols_tab = coef_table(ols_model, 'OLS_HC3')
poisson_tab = coef_table(poisson_model, 'Poisson')

nb_tab = None
if nb_model is not None:
    nb_tab = coef_table(nb_model, 'NegBin')

# Save results
results = {
    'n': int(len(_df)),
    'ols_formula': ols_formula,
    'poisson_formula': poisson_formula,
    'pearson_ratio': float(pearson_ratio),
}

# Write coefficient tables
ols_tab.to_csv('ols_coeffs.csv')
poisson_tab.to_csv('poisson_coeffs.csv')
if nb_tab is not None:
    nb_tab.to_csv('negbin_coeffs.csv')

# Save model fit stats
with open('model_stats.txt', 'w') as f:
    f.write(f"OLS R2: {ols_model.rsquared}\n")
    f.write(f"Poisson AIC: {poisson_model.aic}\n")
    f.write(f"Pearson ratio: {pearson_ratio}\n")
    if nb_model is not None:
        f.write(f"NegBin AIC: {nb_model.aic}\n")

# Print concise summary to stdout
print('N', len(_df))
print('Pearson ratio', pearson_ratio)
print('OLS coeffs:')
print(ols_tab.loc[[c for c in ols_tab.index if c.startswith('age') or c.startswith('sex') or c.startswith('help')]])
print('Poisson coeffs:')
print(poisson_tab.loc[[c for c in poisson_tab.index if c.startswith('age') or c.startswith('sex') or c.startswith('help')]])
if nb_tab is not None:
    print('NegBin coeffs:')
    print(nb_tab.loc[[c for c in nb_tab.index if c.startswith('age') or c.startswith('sex') or c.startswith('help')]])
