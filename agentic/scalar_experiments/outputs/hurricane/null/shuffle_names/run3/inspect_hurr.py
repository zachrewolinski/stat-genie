import pandas as pd
import numpy as np

path = 'hurricane.csv'

_df = pd.read_csv(path)
print(_df.head())
print('\ncolumns', _df.columns.tolist())
print('\nsummary:')
print(_df.describe(include='all').transpose())
