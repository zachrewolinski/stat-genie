import pandas as pd
import numpy as np

df = pd.read_csv('affairs.csv')
age = df['age']

print('unique exact', age.nunique())
print('unique rounded 1', age.round(1).nunique())
print('unique rounded 0', age.round(0).nunique())
print('rounded 0 value counts top')
print(age.round(0).value_counts().head(10))

# show quantiles
print(age.quantile([0,0.1,0.25,0.5,0.75,0.9,1]))

# check if age seems normal
print('mean', age.mean(), 'std', age.std())

