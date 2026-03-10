import pandas as pd

df = pd.read_csv('soccer.csv')

for col in ['meanExp','yellowCards']:
    cond = (df[col] <= df['yellowReds']).mean()
    print(col, '<= yellowReds fraction', cond)

# also check relative to player maybe goals
for col in ['meanExp','yellowCards']:
    cond = (df[col] <= df['player']).mean()
    print(col, '<= player fraction', cond)
