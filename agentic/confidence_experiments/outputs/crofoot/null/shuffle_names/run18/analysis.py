import pandas as pd
import numpy as np

df = pd.read_csv('crofoot.csv')
print('shape', df.shape)
print(df.dtypes)
print('\nunique counts')
print(df.nunique())
print('\nhead')
print(df.head())
print('\nvalue counts for m_focal')
print(df['m_focal'].value_counts())
print('\nvalue counts for win')
print(df['win'].value_counts().sort_index())
print('\nsummary ranges')
for col in df.columns:
    print(col, df[col].min(), df[col].max())
