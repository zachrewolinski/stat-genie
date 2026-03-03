import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')
print('corr age vs pop', df['age'].corr(df['pop']))
print('corr age vs num_amtl', df['age'].corr(df['num_amtl']))
print('corr age vs genus', df['age'].corr(df['genus']))
print('corr pop vs num_amtl', df['pop'].corr(df['num_amtl']))
print('corr pop vs genus', df['pop'].corr(df['genus']))

# check means by age for pop
print(df.groupby('age')['pop'].mean().head())
