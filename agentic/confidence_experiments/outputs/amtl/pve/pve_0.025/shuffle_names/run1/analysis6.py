import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')

neg = df[df['genus'] < 0].head(10)
print('Negative genus examples:')
print(neg[['genus','age','sockets','tooth_class','prob_male']])

exceed = df[df['genus'] > df['age']].head(10)
print('\nGenus > age examples:')
print(exceed[['genus','age','sockets','tooth_class','prob_male']])

# Check for num_amtl > age
exceed_num = df[df['num_amtl'] > df['age']].head(10)
print('\nnum_amtl > age examples:')
print(exceed_num[['num_amtl','age','sockets','tooth_class','prob_male']])

