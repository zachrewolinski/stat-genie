import pandas as pd
import numpy as np

pd.set_option('display.max_rows', 10)

path = 'amtl.csv'
df = pd.read_csv(path)

print('Numeric columns summary:')
print(df[['genus','age','pop','num_amtl','stdev_age']].describe())

print('\nCheck stdev_age unique:', sorted(df['stdev_age'].unique()))

# check if genus aligns with integer counts or proportions
print('\nGenus near integers?')
print((df['genus'] % 1).describe())

# See if genus could be missing count by checking if genus <= age (sockets) 
print('\nProportion of rows where genus <= age:', (df['genus'] <= df['age']).mean())
print('Min/Max genus-age:', (df['genus']-df['age']).min(), (df['genus']-df['age']).max())

# Check if num_amtl could be sockets overall by seeing if num_amtl <= 32
print('\nnum_amtl min/max:', df['num_amtl'].min(), df['num_amtl'].max())

# Check if age (sockets) aligns with tooth class typical counts
print('\nSockets (age) by tooth class:')
print(df.groupby('sockets')['age'].describe())

# See if genus is associated with sockets counts (maybe counts, not transformed)
print('\nGenus by sockets:')
print(df.groupby('sockets')['genus'].describe())

# See if genus values negative exist and how many
print('\nNegative genus count:', (df['genus'] < 0).sum())

# See if genus seems like AMTL proportion by dividing by age (sockets)
df['genus_over_age'] = df['genus'] / df['age']
print('\nGenus/age summary:')
print(df['genus_over_age'].describe())
print('Proportion >1:', (df['genus_over_age']>1).mean())

# maybe num_amtl is sockets total? check num_amtl by tooth_class
print('\nnum_amtl by sockets (tooth class):')
print(df.groupby('sockets')['num_amtl'].describe())

# unique counts for age (sockets) by genus (tooth_class)
print('\nAge (sockets) by tooth_class (actual genus):')
print(df.groupby('tooth_class')['age'].describe())

