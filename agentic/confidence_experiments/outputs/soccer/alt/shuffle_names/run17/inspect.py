import pandas as pd
import numpy as np

df = pd.read_csv('soccer.csv')
print(df[['rater1','nExp']].head())
print('corr', df['rater1'].corr(df['nExp']))
print('unique rater1', sorted(df['rater1'].unique())[:10])
print('unique nExp', sorted(df['nExp'].unique())[:10])
print('counts rater1', df['rater1'].value_counts().sort_index())
print('counts nExp', df['nExp'].value_counts().sort_index())

