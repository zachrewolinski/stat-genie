import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')

# compute ratios
ratio = df['genus'] / df['age']
print('genus/age stats', ratio.describe())

# compare with num_amtl
print('corr(genus/age, num_amtl)', ratio.corr(df['num_amtl']))

# maybe num_amtl is sockets? check correlation with age
print('corr(num_amtl, age)', df['num_amtl'].corr(df['age']))

# check if num_amtl approximately equals pop? etc
print('corr(num_amtl, pop)', df['num_amtl'].corr(df['pop']))

# check if pop maybe equals age + something
print('corr(pop, age)', df['pop'].corr(df['age']))
