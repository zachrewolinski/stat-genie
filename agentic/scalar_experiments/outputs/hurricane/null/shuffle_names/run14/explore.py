import pandas as pd
import numpy as np

df = pd.read_csv('hurricane.csv')
print(df.head())
print(df.dtypes)
print(df.describe(include='all'))
