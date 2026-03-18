import pandas as pd
import numpy as np

df=pd.read_csv('affairs.csv')
idx = pd.Series(range(len(df)))
for col in ['education','age']:
    corr = idx.corr(df[col])
    print(col, 'corr with index', corr)
