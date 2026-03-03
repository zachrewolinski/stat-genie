import pandas as pd
import json

# Load data
amtl = pd.read_csv('amtl.csv')

print('columns', amtl.columns.tolist())
print('shape', amtl.shape)
print(amtl.head())

# summary stats
print(amtl.describe(include='all'))

