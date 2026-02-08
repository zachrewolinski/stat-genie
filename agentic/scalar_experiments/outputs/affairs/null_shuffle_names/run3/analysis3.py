import pandas as pd

df = pd.read_csv('affairs.csv')
affairs = df['age']
children = df['religiousness']
any_affair = (affairs > 0).astype(int)

df2 = df.copy()
df2['any_affair'] = any_affair

prop = df2.groupby('religiousness')['any_affair'].mean()
print(prop)
print('counts')
print(df2.groupby('religiousness')['any_affair'].sum())
print(df2.groupby('religiousness')['any_affair'].count())
