import pandas as pd

df = pd.read_csv('amtl.csv')
print(pd.crosstab(df['feature4'], df['feature1']))
print('feature4 summary by tooth class')
print(df.groupby('feature1')['feature4'].describe())

# check feature4 by genus
print(df.groupby('feature8')['feature4'].describe())

