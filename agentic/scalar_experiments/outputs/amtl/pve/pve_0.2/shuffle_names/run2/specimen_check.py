import pandas as pd

df = pd.read_csv('amtl.csv')

spec = df['prob_male'].iloc[0]
print('Specimen', spec)
print(df[df['prob_male']==spec][['sockets','age','pop','num_amtl','genus','stdev_age','tooth_class','specimen']])

spec2 = df['prob_male'].iloc[10]
print('\nSpecimen2', spec2)
print(df[df['prob_male']==spec2][['sockets','age','pop','num_amtl','genus','stdev_age','tooth_class','specimen']])
