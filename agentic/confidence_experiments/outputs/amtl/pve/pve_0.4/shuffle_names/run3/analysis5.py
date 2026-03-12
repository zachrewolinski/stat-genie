import pandas as pd

df = pd.read_csv('amtl.csv')
print(df.groupby('tooth_class')['genus'].describe())
print(df.groupby('tooth_class')['num_amtl'].describe())
print(df.groupby('tooth_class')['age'].describe())
print(df.groupby('tooth_class')['pop'].describe())
print(df.groupby('tooth_class')['stdev_age'].describe())
