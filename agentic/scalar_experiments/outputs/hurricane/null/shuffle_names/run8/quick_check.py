import pandas as pd
import numpy as np

df=pd.read_csv('hurricane.csv')
print('corr(category, ind)=', df['category'].corr(df['ind']))
print('corr(category, masfem_mturk)=', df['category'].corr(df['masfem_mturk']))
print('corr(ind, masfem_mturk)=', df['ind'].corr(df['masfem_mturk']))

print(df[['category','ind','masfem_mturk']].describe())

