import pandas as pd


df = pd.read_csv('soccer.csv')

for c in ['yellowReds','meanExp','yellowCards']:
    print(c, 'sum', df[c].sum(), 'mean', df[c].mean(), 'max', df[c].max())
