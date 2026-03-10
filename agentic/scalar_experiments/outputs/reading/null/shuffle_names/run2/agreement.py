import pandas as pd
import numpy as np

df = pd.read_csv('reading.csv')
mask = df['device'].notna() & df['correct_rate'].notna()
print('agreement rate', ( (df.loc[mask,'device']>0) == (df.loc[mask,'correct_rate']>0) ).mean())
print(pd.crosstab(df.loc[mask,'device']>0, df.loc[mask,'correct_rate']>0))
