import pandas as pd

df = pd.read_csv('mortgage.csv')
print('corr self_employed vs deny', df[['self_employed','deny']].corr().iloc[0,1])
print('unique combos', df.groupby(['self_employed','deny']).size())
