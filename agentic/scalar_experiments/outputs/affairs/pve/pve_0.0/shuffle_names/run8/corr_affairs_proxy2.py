import pandas as pd
import numpy as np

df = pd.read_csv('affairs.csv')
vals = df['education'] / 1000.0
print('corr edu/1000 vs marital rating (affairs col)', vals.corr(df['affairs']))
print('corr edu/1000 vs religiousness scale (rating col)', vals.corr(df['rating']))
print('corr edu/1000 vs children yes/no', vals.corr(df['religiousness'].map({'yes':1,'no':0})))
