import pandas as pd


df = pd.read_csv('mortgage.csv')
ct = pd.crosstab(df['accept'], df['deny'])
print(ct)
