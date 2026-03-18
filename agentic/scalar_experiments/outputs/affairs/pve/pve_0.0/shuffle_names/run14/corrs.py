import pandas as pd

import numpy as np

df=pd.read_csv('affairs.csv')

num_cols=df.select_dtypes(include=['number']).columns
corr=df[num_cols].corr()
print(corr)
