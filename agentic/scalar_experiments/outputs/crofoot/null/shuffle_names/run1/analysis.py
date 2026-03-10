import pandas as pd
import numpy as np

path = 'crofoot.csv'
df = pd.read_csv(path)
print(df.head())
print(df.describe(include='all'))
print('dtypes:', df.dtypes)
print('unique counts:', df.nunique())

