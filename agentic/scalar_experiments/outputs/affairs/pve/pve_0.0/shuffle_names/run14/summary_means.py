import pandas as pd

df=pd.read_csv('affairs.csv')
for col in ['rating','affairs']:
    print(col, df[col].mean(), df[col].value_counts().sort_index())
