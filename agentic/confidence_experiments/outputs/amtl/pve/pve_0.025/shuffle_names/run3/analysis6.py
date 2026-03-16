import pandas as pd

df = pd.read_csv('amtl.csv')
# pick first specimen
spec = df['prob_male'].iloc[0]
print('Specimen', spec)
print(df[df['prob_male']==spec][['sockets','genus','age','pop','num_amtl','stdev_age','tooth_class','specimen']])

# check ranges of pop and num_amtl by genus
print('\nPop range by genus:')
print(df.groupby('tooth_class')['pop'].agg(['min','max','mean']))
print('\nnum_amtl range by genus:')
print(df.groupby('tooth_class')['num_amtl'].agg(['min','max','mean']))
