import pandas as pd


df = pd.read_csv('panda_nuts.csv')
df['efficiency'] = df['nuts_opened'] / df['seconds']

print('mean efficiency overall', df['efficiency'].mean())
print('\nmean efficiency by sex')
print(df.groupby('sex')['efficiency'].agg(['mean','median','count']))
print('\nmean efficiency by help')
print(df.groupby('help')['efficiency'].agg(['mean','median','count']))

# correlation with age
print('\ncorrelation age vs efficiency')
print(df[['age','efficiency']].corr().iloc[0,1])
