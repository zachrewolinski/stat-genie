import pandas as pd

path='soccer.csv'
df=pd.read_csv(path)

for col in ['yellowCards','meanExp','yellowReds']:
    print('\n', col)
    print(df[col].value_counts().sort_index().head(10))
    print('mean', df[col].mean(), 'max', df[col].max())
