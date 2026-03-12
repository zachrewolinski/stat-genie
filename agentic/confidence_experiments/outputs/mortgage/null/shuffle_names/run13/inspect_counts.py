import pandas as pd

df = pd.read_csv('mortgage.csv')
print('deny mean', df['deny'].mean())
print('accept mean', df['accept'].mean())
print('deny unique', df['deny'].unique()[:10])
print('accept unique', df['accept'].unique()[:10])
print('accept+deny min/max', (df['accept']+df['deny']).min(), (df['accept']+df['deny']).max())
print('crosstab deny vs accept')
print(pd.crosstab(df['deny'], df['accept']))
print('corr')
print(df[['deny','accept']].corr())
print('female counts')
print(df['female'].value_counts())
