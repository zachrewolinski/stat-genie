import pandas as pd
import json
import numpy as np

# Load data
_df = pd.read_csv('amtl.csv')
print(_df.head())
print(_df.dtypes)
print(_df.describe(include='all').transpose().head(15))
print('rows', len(_df))
print(_df['feature3'].describe())
print(_df['feature4'].describe())
print(_df['feature8'].value_counts())
