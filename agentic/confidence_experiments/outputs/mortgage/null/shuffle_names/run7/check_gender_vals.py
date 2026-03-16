import pandas as pd

df = pd.read_csv('mortgage.csv')
print(df['denied_PMI'].unique()[:10])
print(df['denied_PMI'].value_counts(dropna=False))
