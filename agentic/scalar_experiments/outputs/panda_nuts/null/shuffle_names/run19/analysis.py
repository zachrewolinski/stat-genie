import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


df = pd.read_csv('panda_nuts.csv')

# Infer column meanings
print('head')
print(df.head())
print('dtypes')
print(df.dtypes)

# basic stats
print('age range', df['age'].min(), df['age'].max(), df['age'].nunique())
print('hammer range', df['hammer'].min(), df['hammer'].max(), df['hammer'].nunique())
print('nuts_opened unique', df['nuts_opened'].unique())
print('sex unique', df['sex'].unique())
print('help range', df['help'].min(), df['help'].max())
print('chimpanzee range', df['chimpanzee'].min(), df['chimpanzee'].max())
print('seconds unique', df['seconds'].unique())

# Create renamed columns
renamed = df.rename(columns={
    'nuts_opened':'sex_actual',
    'sex':'hammer_type',
    'help':'nuts_opened_actual',
    'chimpanzee':'seconds_actual',
    'seconds':'help_received'
})

# Compute efficiency (nuts per second)
renamed['efficiency'] = renamed['nuts_opened_actual'] / renamed['seconds_actual']

# Check for zero seconds
print('zero seconds', (renamed['seconds_actual']<=0).sum())

# Summary by help
print(renamed.groupby('help_received')['efficiency'].describe())

# OLS with age, sex, help
# If age is not actual age maybe use hammer? We'll test both.

# Ensure categories
renamed['sex_actual'] = renamed['sex_actual'].astype('category')
renamed['help_received'] = renamed['help_received'].astype('category')

model = smf.ols('efficiency ~ age + C(sex_actual) + C(help_received)', data=renamed).fit()
print(model.summary())

model2 = smf.ols('efficiency ~ hammer + C(sex_actual) + C(help_received)', data=renamed).fit()
print(model2.summary())

# Robust SE
model_robust = model.get_robustcov_results(cov_type='HC3')
print(model_robust.summary())

