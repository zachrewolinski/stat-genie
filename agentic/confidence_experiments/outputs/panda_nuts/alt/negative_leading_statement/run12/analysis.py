import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm


df = pd.read_csv('panda_nuts.csv')

# Efficiency: nuts opened per minute (for interpretability)
df['minutes'] = df['seconds'] / 60.0
df['rate_per_min'] = df['nuts_opened'] / df['minutes']

# Basic summaries
n = len(df)
rate_summary = df['rate_per_min'].describe()

# Group summaries
sex_summary = df.groupby('sex')['rate_per_min'].agg(['count','mean','median','std'])
help_summary = df.groupby('help')['rate_per_min'].agg(['count','mean','median','std'])

# Poisson GLM with offset log(seconds)
poisson_model = smf.glm(
    formula='nuts_opened ~ age + C(sex) + C(help)',
    data=df,
    family=sm.families.Poisson(),
    offset=np.log(df['seconds'])
)
poisson_res = poisson_model.fit(cov_type='HC0')

# Overdispersion check
pearson_chi2 = np.sum(poisson_res.resid_pearson**2)
overdispersion = pearson_chi2 / poisson_res.df_resid

# Negative binomial with offset (accounts for overdispersion)
nb_model = smf.negativebinomial(
    formula='nuts_opened ~ age + C(sex) + C(help)',
    data=df,
    offset=np.log(df['seconds'])
)
nb_res = nb_model.fit(disp=0)

# Rate ratios and CI for NB
nb_params = nb_res.params
nb_conf = nb_res.conf_int()
nb_rate_ratios = np.exp(nb_params)
nb_rr_ci = np.exp(nb_conf)

# OLS on rate per minute for robustness (approximate)
ols_model = smf.ols('rate_per_min ~ age + C(sex) + C(help)', data=df)
ols_res = ols_model.fit(cov_type='HC3')

print('N:', n)
print('\nRate per minute summary:')
print(rate_summary)
print('\nRate per minute by sex:')
print(sex_summary)
print('\nRate per minute by help:')
print(help_summary)

print('\nPoisson GLM (robust SE) summary:')
print(poisson_res.summary())
print('\nOverdispersion (Pearson chi2/df):', overdispersion)

print('\nNegative Binomial (offset) summary:')
print(nb_res.summary())
print('\nNegative Binomial rate ratios (exp coef) and 95% CI:')
nb_rr_table = pd.DataFrame({
    'rate_ratio': nb_rate_ratios,
    'ci_low': nb_rr_ci[0],
    'ci_high': nb_rr_ci[1],
    'p_value': nb_res.pvalues
})
print(nb_rr_table)

print('\nOLS on rate per minute (robust SE) summary:')
print(ols_res.summary())
