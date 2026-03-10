import pandas as pd
import numpy as np

path = 'panda_nuts.csv'
df = pd.read_csv(path)
print(df.head())
print(df.dtypes)
print(df['nuts_opened'].unique()[:10])
print(df['sex'].unique())
print(df['seconds'].unique())
print(df[['age','hammer','help','chimpanzee']].describe())
print('rows', len(df))

# Check correlations
# infer mapping
# propose: sex column actual hammer type; nuts_opened actual sex; help actual nuts opened; chimpanzee actual seconds; seconds actual help (Y/N)

# compute efficiency = nuts opened per second

df2 = df.copy()
df2['sex_actual'] = df2['nuts_opened']
df2['hammer_type'] = df2['sex']
df2['nuts_opened_actual'] = df2['help']
df2['seconds_actual'] = df2['chimpanzee']
df2['help_actual'] = df2['seconds'].str.upper().map({'Y':1,'N':0})

# drop any zero seconds

df2['efficiency'] = df2['nuts_opened_actual'] / df2['seconds_actual']
print(df2[['efficiency']].describe())

# basic group means
print(df2.groupby('sex_actual')['efficiency'].mean())
print(df2.groupby('help_actual')['efficiency'].mean())

# regression: efficiency ~ age + sex + help
import statsmodels.api as sm
import statsmodels.formula.api as smf

model = smf.ols('efficiency ~ age + C(sex_actual) + help_actual', data=df2).fit()
print(model.summary())

# Also consider log efficiency if skewed
model_log = smf.ols('np.log1p(efficiency) ~ age + C(sex_actual) + help_actual', data=df2).fit()
print(model_log.summary())

