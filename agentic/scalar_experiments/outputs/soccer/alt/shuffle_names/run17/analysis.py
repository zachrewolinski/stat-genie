import pandas as pd
import numpy as np

path = 'soccer.csv'
df = pd.read_csv(path)
print(df.head())
print(df.shape)
print(df.dtypes)

# numeric summary
num_cols = df.select_dtypes(include=[np.number]).columns
print('num cols', num_cols)
print(df[num_cols].describe().T[['mean','std','min','max']].head(10))

# show unique counts for all
print('nunique:')
print(df.nunique().sort_values())

