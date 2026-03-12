import pandas as pd

df = pd.read_csv('mortgage.csv')
if 'accept' in df.columns:
    y = df['accept']
else:
    y = 1 - df['deny']
sub = df[['female']].copy()
sub['accept']=y
sub = sub[sub['female'].isin([0,1]) & sub['accept'].isin([0,1])]
print(sub.groupby('female')['accept'].agg(['count','sum','mean']))
