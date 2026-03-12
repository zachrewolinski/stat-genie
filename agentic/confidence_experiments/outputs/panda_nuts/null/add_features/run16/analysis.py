import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

_df = pd.read_csv('panda_nuts.csv')

cols = ['nuts_opened', 'seconds', 'age', 'sex', 'help']
df = _df[cols].copy()
for col in ['sex', 'help']:
    df[col] = df[col].astype(str).str.strip().str.lower()

_df_clean = df[(df['seconds'] > 0)].dropna()

_df_clean['rate_per_sec'] = _df_clean['nuts_opened'] / _df_clean['seconds']
_df_clean['rate_per_min'] = _df_clean['rate_per_sec'] * 60

model = smf.glm(
    formula='nuts_opened ~ age + C(sex) + C(help)',
    data=_df_clean,
    family=sm.families.Poisson(),
    offset=np.log(_df_clean['seconds'])
)

res = model.fit()
# Robust SE via fit(cov_type=...)
res_robust = model.fit(cov_type='HC0')

pearson_chi2 = res.pearson_chi2
pearson_df = res.df_resid
overdispersion = pearson_chi2 / pearson_df if pearson_df > 0 else float('nan')

ols = smf.ols('rate_per_sec ~ age + C(sex) + C(help)', data=_df_clean).fit()

print('N:', len(_df_clean))
print('Rate per min summary:')
print(_df_clean['rate_per_min'].describe())
print('\nPoisson GLM results (naive SE):')
print(res.summary())
print('\nPoisson GLM results (robust SE):')
print(res_robust.summary())
print('\nOverdispersion (Pearson chi2/df):', overdispersion)
print('\nOLS results:')
print(ols.summary())

summary = pd.DataFrame({
    'coef': res.params,
    'se': res.bse,
    'pvalue': res.pvalues,
    'rr': np.exp(res.params)
})
summary.index.name = 'term'
summary.to_csv('glm_results.csv')

summary_robust = pd.DataFrame({
    'coef': res_robust.params,
    'se': res_robust.bse,
    'pvalue': res_robust.pvalues,
    'rr': np.exp(res_robust.params)
})
summary_robust.index.name = 'term'
summary_robust.to_csv('glm_results_robust.csv')

ols_summary = pd.DataFrame({
    'coef': ols.params,
    'se': ols.bse,
    'pvalue': ols.pvalues
})
ols_summary.index.name = 'term'
ols_summary.to_csv('ols_results.csv')
