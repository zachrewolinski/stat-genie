import pandas as pd
import numpy as np

path = 'reading.csv'
df = pd.read_csv(path)
print(df.head())
print(df.describe(include='all').transpose().head(25))
print('columns', df.columns.tolist())
