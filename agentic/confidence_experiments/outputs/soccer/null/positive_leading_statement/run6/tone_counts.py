import pandas as pd

df = pd.read_csv('soccer.csv')
df['skin_tone'] = df[['rater1','rater2']].mean(axis=1)
print(df['skin_tone'].value_counts(dropna=False).sort_index())
