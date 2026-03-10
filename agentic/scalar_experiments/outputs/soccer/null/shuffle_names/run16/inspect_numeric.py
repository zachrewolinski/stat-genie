import pandas as pd
import numpy as np

path='soccer.csv'
df=pd.read_csv(path)

num_cols=df.select_dtypes(include=[np.number]).columns
print('numeric columns', num_cols)
summary = df[num_cols].describe().T
summary['unique'] = df[num_cols].nunique()
print(summary)
