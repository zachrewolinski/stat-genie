import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('panda_nuts.csv')

# Clean/standardize
_df['sex'] = _df['sex'].astype('category')
_df['help'] = _df['help'].astype('category')
_df['hammer'] = _df['hammer'].astype('category')

# Efficiency rate
_df['efficiency'] = _df['nuts_opened'] / _df['seconds']

print('rows', len(_df))
print(_df.head())
print(_df[['nuts_opened','seconds','efficiency']].describe())
print('help levels', _df['help'].cat.categories)
print('sex levels', _df['sex'].cat.categories)

# GLM Poisson with offset log(seconds)
_df['log_seconds'] = np.log(_df['seconds'])

formula = 'nuts_opened ~ age + C(sex) + C(help) + C(hammer)'
model = smf.glm(formula=formula, data=_df, family=sm.families.Poisson(), offset=_df['log_seconds'])
res = model.fit(cov_type='cluster', cov_kwds={'groups': _df['chimpanzee']})
print(res.summary())

# Overdispersion check (Pearson chi2 / df)
pearson_chi2 = sum(res.resid_pearson**2)
ratio = pearson_chi2 / res.df_resid
print('overdispersion_ratio', ratio)

# Negative binomial if overdispersion
model_nb = smf.glm(formula=formula, data=_df, family=sm.families.NegativeBinomial(alpha=1.0), offset=_df['log_seconds'])
res_nb = model_nb.fit(cov_type='cluster', cov_kwds={'groups': _df['chimpanzee']})
print(res_nb.summary())

# Linear regression on efficiency
lm = smf.ols('efficiency ~ age + C(sex) + C(help) + C(hammer)', data=_df).fit(cov_type='cluster', cov_kwds={'groups': _df['chimpanzee']})
print(lm.summary())

