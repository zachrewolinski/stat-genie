import pandas as pd

df = pd.read_csv('amtl.csv')
print(df.head())
print('\nDtypes:')
print(df.dtypes)
print('\nSummary (numeric):')
print(df.describe())
print('\nSummary (categorical):')
print(df.describe(include=['object']))
print('\nUnique feature1:', df['feature1'].unique())
print('Unique feature8:', df['feature8'].unique())
