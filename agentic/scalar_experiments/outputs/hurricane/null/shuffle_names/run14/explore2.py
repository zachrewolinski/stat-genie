import pandas as pd
import numpy as np

df = pd.read_csv('hurricane.csv')

print('missing counts')
print(df.isna().sum())

for col in df.columns:
    if df[col].dtype != 'object':
        print('\n', col)
        print(df[col].describe())

print('\nunique values for min column:', df['min'].unique())

# show top deaths? if name is deaths
print('\nTop rows by name (potential deaths):')
print(df.sort_values('name', ascending=False).head(10)[['alldeaths','name','wind','year','gender_mf','category','masfem_mturk','ind','masfem','elapsedyrs','source','ndam15']])

# show top rows by elapsedyrs (potential damages)
print('\nTop rows by elapsedyrs:')
print(df.sort_values('elapsedyrs', ascending=False).head(10)[['alldeaths','elapsedyrs','source','name','wind','year']])

# top by source
print('\nTop rows by source:')
print(df.sort_values('source', ascending=False).head(10)[['alldeaths','source','elapsedyrs','name','wind','year']])
