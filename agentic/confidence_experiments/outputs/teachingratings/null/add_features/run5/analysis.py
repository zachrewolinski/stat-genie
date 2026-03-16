import pandas as pd
import numpy as np

df = pd.read_csv('teachingratings.csv')
print(df.head())
print('columns', df.columns.tolist())
print(df.describe(include='all'))
