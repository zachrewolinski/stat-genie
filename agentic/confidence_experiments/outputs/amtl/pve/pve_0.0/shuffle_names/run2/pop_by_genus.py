import pandas as pd

df = pd.read_csv('amtl.csv')

print(df.groupby('tooth_class')['pop'].describe())
print('\nnum_amtl by genus:')
print(df.groupby('tooth_class')['num_amtl'].describe())
