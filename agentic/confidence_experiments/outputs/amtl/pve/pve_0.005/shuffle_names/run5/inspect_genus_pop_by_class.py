import pandas as pd

df = pd.read_csv('amtl.csv')
for c in df['sockets'].unique():
    sub = df[df['sockets']==c]
    print(c, sub['genus'].corr(sub['pop']))
