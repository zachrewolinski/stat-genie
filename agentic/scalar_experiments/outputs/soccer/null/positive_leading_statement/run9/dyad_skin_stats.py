import pandas as pd

df = pd.read_csv('soccer.csv')
df['skin'] = df[['rater1','rater2']].mean(axis=1)

print('dyads with skin:', df['skin'].notna().sum())
print('skin value counts (rounded 2) top 10):')
print(df['skin'].round(2).value_counts().head(10))
print('dark > 0.5 count:', (df['skin'] > 0.5).sum())
print('dark >= 0.5 count:', (df['skin'] >= 0.5).sum())
