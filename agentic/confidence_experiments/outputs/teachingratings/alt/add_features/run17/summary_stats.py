import pandas as pd

df = pd.read_csv('teachingratings.csv')
print('beauty mean', df['beauty'].mean(), 'sd', df['beauty'].std())
print('eval mean', df['eval'].mean(), 'sd', df['eval'].std())
