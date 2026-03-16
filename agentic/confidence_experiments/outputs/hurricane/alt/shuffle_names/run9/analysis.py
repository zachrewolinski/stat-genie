import pandas as pd
import numpy as np


df = pd.read_csv('hurricane.csv')
print(df.head())
print('\nSummary:')
print(df.describe(include='all'))
print('\nUnique counts:')
print(df.nunique())

# quick correlations among numeric columns
num_cols = df.select_dtypes(include=[np.number]).columns
corr = df[num_cols].corr()
print('\nNumeric columns:', list(num_cols))
print('\nCorrelation matrix (rounded):')
print(corr.round(3))
