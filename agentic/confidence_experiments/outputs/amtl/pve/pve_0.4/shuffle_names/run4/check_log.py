import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')

exp_genus = np.exp(df['genus'])
print('exp(genus) range', exp_genus.min(), exp_genus.max(), exp_genus.mean())
print('corr exp_genus with age', exp_genus.corr(df['age']))
print('corr exp_genus with num_amtl', exp_genus.corr(df['num_amtl']))
print('exp_genus <= age proportion', (exp_genus <= df['age']).mean())
