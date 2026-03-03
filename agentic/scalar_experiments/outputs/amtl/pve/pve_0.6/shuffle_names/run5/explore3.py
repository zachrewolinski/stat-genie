import pandas as pd
amtl = pd.read_csv('amtl.csv')
print(amtl.groupby('tooth_class')[['genus','age','pop','num_amtl','stdev_age']].mean())
print('\nstds')
print(amtl.groupby('tooth_class')[['genus','age','pop','num_amtl','stdev_age']].std())
