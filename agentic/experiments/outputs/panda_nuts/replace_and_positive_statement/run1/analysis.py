import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('panda_nuts.csv')

# Nut-cracking efficiency: nuts opened per second
_df['efficiency'] = _df['nuts_opened'] / _df['seconds']

# Basic model: efficiency ~ age + sex + help
model = smf.ols('efficiency ~ age + C(sex) + C(help)', data=_df).fit()

# Alternative: log-transformed efficiency for robustness
model_log = smf.ols('np.log1p(efficiency) ~ age + C(sex) + C(help)', data=_df).fit()

# Optional control for hammer type (not required by question, used as sensitivity check)
model_hammer = smf.ols('efficiency ~ age + C(sex) + C(help) + C(hammer)', data=_df).fit()

print('Efficiency summary:')
print(_df['efficiency'].describe())
print('\nOLS: efficiency ~ age + sex + help')
print(model.summary())
print('\nOLS: log1p(efficiency) ~ age + sex + help')
print(model_log.summary())
print('\nOLS with hammer control')
print(model_hammer.summary())
