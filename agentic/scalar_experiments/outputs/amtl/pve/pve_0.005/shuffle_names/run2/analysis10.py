import pandas as pd

df = pd.read_csv('amtl.csv')

for tc in df['sockets'].unique():
    sub = df[df['sockets']==tc]
    corr = sub['genus'].corr(sub['age'])
    print(tc, 'corr genus vs age', corr)

