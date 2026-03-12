import pandas as pd

df = pd.read_csv('mortgage.csv')
ct = pd.crosstab(df['denied_PMI'], df['deny'])
print(ct)
