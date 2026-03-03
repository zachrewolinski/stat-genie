import pandas as pd


df = pd.read_csv('amtl.csv')

num_cols = ['genus','age','pop','num_amtl','stdev_age']
for col in num_cols:
    print('\n', col)
    print(df.groupby('tooth_class')[col].describe()[['count','mean','std','min','max']])

print('\nBy sockets (tooth class) for numeric cols')
for col in num_cols:
    print('\n', col)
    print(df.groupby('sockets')[col].describe()[['count','mean','std','min','max']])

