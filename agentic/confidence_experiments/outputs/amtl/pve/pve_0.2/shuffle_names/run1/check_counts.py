import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')

# Check if genus could be count of missing teeth by verifying 0<=genus<=age and near-integer
cond_genus = (df['genus'] >= 0) & (df['genus'] <= df['age'])
cond_num = (df['num_amtl'] >= 0) & (df['num_amtl'] <= df['age'])
print('genus within [0, age]:', cond_genus.mean())
print('num_amtl within [0, age]:', cond_num.mean())

# Near integer checks
near_int_genus = np.isclose(df['genus'], df['genus'].round(), atol=0.01)
near_int_num = np.isclose(df['num_amtl'], df['num_amtl'].round(), atol=0.01)
print('genus near integer fraction', near_int_genus.mean())
print('num_amtl near integer fraction', near_int_num.mean())

# compare num_amtl with age for specimen-level? maybe num_amtl is sockets
cond_num_vs_age = (df['age'] >= 0) & (df['age'] <= df['num_amtl'])
print('age within [0, num_amtl]:', cond_num_vs_age.mean())

# Count zeros
print('genus zeros', (df['genus'] == 0).sum())
print('num_amtl zeros', (df['num_amtl'] == 0).sum())

