import pandas as pd

path='soccer.csv'
df=pd.read_csv(path)

for col in ['meanExp','yellowCards']:
    corr = df[col].corr(df['yellowReds'])
    print(col, 'corr with yellowReds', corr)
