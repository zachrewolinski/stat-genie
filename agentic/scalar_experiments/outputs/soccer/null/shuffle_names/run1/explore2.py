import pandas as pd

df = pd.read_csv('soccer.csv')

for col in ['yellowReds','yellowCards','redCards']:
    print(col, df[col].describe())
    print('unique sample', sorted(df[col].unique())[:10])

