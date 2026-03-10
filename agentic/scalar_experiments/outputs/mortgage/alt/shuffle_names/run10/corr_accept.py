import pandas as pd

df = pd.read_csv('mortgage.csv')
print('corr accept vs deny', df[['accept','deny']].corr().iloc[0,1])
print('corr accept vs self_employed', df[['accept','self_employed']].corr().iloc[0,1])
print(df.groupby(['accept','deny']).size())
