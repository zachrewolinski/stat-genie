import pandas as pd

df = pd.read_csv('soccer.csv')

for col in ['meanExp','yellowCards','yellowReds']:
    s = df[col]
    print(col, 'sum', s.sum(), 'mean', s.mean(), 'max', s.max())
