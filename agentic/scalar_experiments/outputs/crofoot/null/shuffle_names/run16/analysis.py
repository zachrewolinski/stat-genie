import pandas as pd
import numpy as np

df = pd.read_csv('crofoot.csv')
print(df.head())
print('\nsummary:')
print(df.describe().T[['min','max','mean','std','count']])
print('\nunique counts:')
print(df.nunique())
