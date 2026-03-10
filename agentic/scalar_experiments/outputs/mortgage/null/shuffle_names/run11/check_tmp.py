import pandas as pd

df = pd.read_csv('mortgage.csv')
print('accept+deny unique', (df['accept']+df['deny']).unique()[:10])
print('accept==1 & deny==1', ((df['accept']==1) & (df['deny']==1)).sum())
print('accept==0 & deny==0', ((df['accept']==0) & (df['deny']==0)).sum())
print('accept==1 & deny==0', ((df['accept']==1) & (df['deny']==0)).sum())
print('accept==0 & deny==1', ((df['accept']==0) & (df['deny']==1)).sum())
