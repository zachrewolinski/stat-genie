import pandas as pd

df = pd.read_csv('soccer.csv')

games = df['redCards']
for col in ['meanExp','yellowCards','yellowReds']:
    print(col, 'corr with games', df[col].corr(games))
