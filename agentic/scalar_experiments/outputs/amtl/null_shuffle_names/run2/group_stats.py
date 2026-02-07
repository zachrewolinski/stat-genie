import pandas as pd

df = pd.read_csv('amtl.csv')

cols = ['sockets','genus','age','pop','num_amtl','stdev_age','tooth_class','specimen']
for col in cols:
    nunique = df.groupby('prob_male')[col].nunique()
    frac_const = (nunique==1).mean()
    print(col, 'frac_const', frac_const, 'nunique_counts', nunique.value_counts().to_dict())
