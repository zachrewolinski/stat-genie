import pandas as pd
import numpy as np

_df = pd.read_csv('hurricane.csv')
print(_df.head())
print(_df.columns)
print(_df[['masfem','gender_mf','alldeaths','wind','min','category']].describe())
print('missing', _df[['masfem','gender_mf','alldeaths','wind','min','category']].isna().sum())
