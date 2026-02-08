import pandas as pd

df = pd.read_csv('amtl.csv')
cols = ['sockets','genus','age','pop','num_amtl','stdev_age','tooth_class','specimen']
for col in cols:
    nunique_per = df.groupby('prob_male')[col].nunique()
    print(col, 'prop constant', (nunique_per==1).mean())
