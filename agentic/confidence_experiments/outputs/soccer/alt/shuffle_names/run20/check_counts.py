import pandas as pd


df = pd.read_csv('soccer.csv')

for c in ['meanExp','yellowCards']:
    print(c, 'sum', df[c].sum(), 'mean', df[c].mean())
