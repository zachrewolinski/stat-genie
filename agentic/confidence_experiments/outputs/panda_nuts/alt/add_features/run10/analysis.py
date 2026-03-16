import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('panda_nuts.csv')

# Clean / prep
_df = _df[_df['seconds'] > 0].copy()
_df['rate'] = _df['nuts_opened'] / _df['seconds']

# Recode help to clean string
_df['help'] = _df['help'].astype(str).str.strip()

# Model: Poisson with offset for time (counts over exposure)
_df['log_seconds'] = np.log(_df['seconds'])

formula = 'nuts_opened ~ age + C(sex) + C(help)'

model = smf.glm(formula=formula, data=_df, family=sm.families.Poisson(), offset=_df['log_seconds'])
res = model.fit(cov_type='cluster', cov_kwds={'groups': _df['chimpanzee']})

# Also compute linear model on log rate for interpretability
# Avoid log(0) by adding small constant
_df['log_rate'] = np.log(_df['rate'] + 1e-6)
ols = smf.ols('log_rate ~ age + C(sex) + C(help)', data=_df).fit(cov_type='cluster', cov_kwds={'groups': _df['chimpanzee']})

# Descriptives
_desc = _df.groupby(['sex', 'help']).agg(
    n=('rate', 'size'),
    mean_rate=('rate', 'mean'),
    median_rate=('rate', 'median'),
)

print('N rows:', _df.shape[0])
print('Unique chimps:', _df['chimpanzee'].nunique())
print('\nPoisson (rate with offset) clustered by chimpanzee:')
print(res.summary())
print('\nOLS log-rate clustered by chimpanzee:')
print(ols.summary())
print('\nDescriptives by sex/help:')
print(_desc)

# Save key results for later
out = {
    'poisson_params': res.params.to_dict(),
    'poisson_pvalues': res.pvalues.to_dict(),
    'poisson_ci': res.conf_int().to_dict(),
    'ols_params': ols.params.to_dict(),
    'ols_pvalues': ols.pvalues.to_dict(),
    'ols_ci': ols.conf_int().to_dict(),
}

pd.Series(out).to_json('analysis_results.json')
