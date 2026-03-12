import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')

missing = np.exp(df['genus'])

# check if missing <= age for most rows
ratio = (missing <= df['age']).mean()
print('Fraction missing<=age:', ratio)
print('Max missing-age', (missing - df['age']).max())
print('Percent rows missing>age', (missing > df['age']).mean())

# show some rows where missing>age
print(df.loc[(missing>df['age'])].head())
