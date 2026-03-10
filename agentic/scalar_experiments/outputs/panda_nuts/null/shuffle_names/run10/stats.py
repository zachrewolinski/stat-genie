import pandas as pd
import statsmodels.formula.api as smf

# Load data
raw = pd.read_csv('panda_nuts.csv')

# Map columns based on inspection and metadata shuffle
# Actual variables inferred:
# age -> age (years)
# nuts_opened -> sex (m/f)
# seconds -> help (y/N)
# help -> nuts_opened (count)
# chimpanzee -> seconds (duration)

# Build cleaned dataframe

df = pd.DataFrame({
    'age': raw['age'],
    'sex': raw['nuts_opened'].astype(str),
    'helped': raw['seconds'].astype(str).str.lower().map({'y': 1, 'n': 0}),
    'nuts_opened': raw['help'].astype(float),
    'seconds': raw['chimpanzee'].astype(float),
    'chimp_id': raw['hammer']
})

# Efficiency
# Avoid divide-by-zero (none expected)
df['efficiency'] = df['nuts_opened'] / df['seconds']

print('Efficiency summary')
print(df['efficiency'].describe())
print('\nHelp counts')
print(df['helped'].value_counts(dropna=False))
print('\nSex counts')
print(df['sex'].value_counts(dropna=False))

# OLS regression
model = smf.ols('efficiency ~ age + C(sex) + helped', data=df).fit()
print('\nOLS summary:')
print(model.summary())

# Also check log efficiency as robustness (add small constant if needed)
import numpy as np

df['log_eff'] = np.log(df['efficiency'] + 1e-6)
model_log = smf.ols('log_eff ~ age + C(sex) + helped', data=df).fit()
print('\nLog OLS summary:')
print(model_log.summary())
