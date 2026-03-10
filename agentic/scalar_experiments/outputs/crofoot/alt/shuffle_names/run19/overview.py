import pandas as pd
import numpy as np

pd.set_option('display.width', 120)
pd.set_option('display.max_columns', None)

df = pd.read_csv('crofoot.csv')
print(df.head())
print('\nDescribe:')
print(df.describe())
print('\nNunique:')
print(df.nunique())
print('\nDtypes:')
print(df.dtypes)
