import pandas as pd
import numpy as np


df = pd.read_csv('affairs.csv')
for col in df.columns:
    if df[col].dtype == 'object' or df[col].nunique() <= 10:
        print('\n', col, 'nunique', df[col].nunique())
        print(sorted(df[col].unique()))
    else:
        print('\n', col, 'nunique', df[col].nunique(), 'min', df[col].min(), 'max', df[col].max(), 'mean', df[col].mean())
