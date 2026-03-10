import pandas as pd
import numpy as np

path = 'hurricane.csv'
df = pd.read_csv(path)
print(df.head())
print(df.columns)
print(df.describe(include='all'))
