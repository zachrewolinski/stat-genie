import pandas as pd
import numpy as np

path = 'affairs.csv'
df = pd.read_csv(path)
print(df.head())
print(df.describe(include='all'))
print(df.nunique())
