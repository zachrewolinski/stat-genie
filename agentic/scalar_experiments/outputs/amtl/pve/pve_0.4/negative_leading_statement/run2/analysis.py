import pandas as pd
import numpy as np

# Load data
amtl = pd.read_csv('amtl.csv')
print(amtl.head())
print(amtl.dtypes)
print(amtl.describe(include='all'))
print('rows', len(amtl))
print('unique genus', amtl['genus'].unique())
print('num_amtl min max', amtl['num_amtl'].min(), amtl['num_amtl'].max())
