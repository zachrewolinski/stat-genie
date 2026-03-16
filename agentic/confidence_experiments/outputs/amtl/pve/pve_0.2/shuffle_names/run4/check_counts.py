import pandas as pd
import numpy as np

df=pd.read_csv('amtl.csv')

# check if genus could be count <= age
print('genus negative count', (df['genus']<0).sum())
print('genus > age count', (df['genus']>df['age']).sum())

# check if num_amtl <= age
print('num_amtl negative count', (df['num_amtl']<0).sum())
print('num_amtl > age count', (df['num_amtl']>df['age']).sum())

# check if num_amtl <= pop (age?)
print('num_amtl > pop count', (df['num_amtl']>df['pop']).sum())

# check if genus between 0 and age if shift
print('genus range vs age min max:', df['genus'].min(), df['genus'].max(), df['age'].min(), df['age'].max())
