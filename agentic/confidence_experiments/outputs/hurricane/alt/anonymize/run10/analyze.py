import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


df = pd.read_csv('hurricane.csv')
print(df.head())
print(df.describe(include='all'))

# Rename columns for clarity
cols = {f'feature{i}': f'f{i}' for i in range(1, 15)}
df = df.rename(columns=cols)

# Key variables
# f4: femininity index (1-11)
# f6: gender indicator (0 male, 1 female)
# f8: deaths
# f7: category
# f5: min pressure
# f13: max wind speed
# f9/f14: damages (normalized)

# transform deaths

df['log_deaths'] = np.log1p(df['f8'])

# basic correlations
print('\nCorrelations with femininity (f4):')
for col in ['f8', 'log_deaths', 'f7', 'f5', 'f13', 'f9', 'f14']:
    corr = df[['f4', col]].corr().iloc[0,1]
    print(col, corr)

# simple regression: log_deaths ~ f4
model1 = smf.ols('log_deaths ~ f4', data=df).fit()
print('\nModel1 log_deaths ~ f4')
print(model1.summary())

# controlling for intensity (category, pressure, wind)
model2 = smf.ols('log_deaths ~ f4 + f7 + f5 + f13', data=df).fit()
print('\nModel2 log_deaths ~ f4 + f7 + f5 + f13')
print(model2.summary())

# using gender indicator
model3 = smf.ols('log_deaths ~ f6', data=df).fit()
print('\nModel3 log_deaths ~ f6')
print(model3.summary())

model4 = smf.ols('log_deaths ~ f6 + f7 + f5 + f13', data=df).fit()
print('\nModel4 log_deaths ~ f6 + f7 + f5 + f13')
print(model4.summary())

# alternative: Poisson for deaths (count)
model5 = smf.glm('f8 ~ f4 + f7 + f5 + f13', data=df, family=sm.families.Poisson()).fit()
print('\nPoisson model deaths ~ f4 + f7 + f5 + f13')
print(model5.summary())

# check interaction: femininity * severity (category)
model6 = smf.ols('log_deaths ~ f4 * f7 + f5 + f13', data=df).fit()
print('\nModel6 log_deaths ~ f4*f7 + f5 + f13')
print(model6.summary())

# using alternative femininity (f12)
model7 = smf.ols('log_deaths ~ f12 + f7 + f5 + f13', data=df).fit()
print('\nModel7 log_deaths ~ f12 + f7 + f5 + f13')
print(model7.summary())

# maybe include damage? But damage may be outcome too. We'll just see correlation

