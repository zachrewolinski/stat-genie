import pandas as pd
import numpy as np
from pathlib import Path

df = pd.read_csv('affairs.csv')
print(df.head())
print(df.describe(include='all'))
print(df.dtypes)

# Show unique counts and min/max for numeric columns
for col in df.columns:
    print('\n', col)
    if df[col].dtype == 'object':
        print(df[col].value_counts())
    else:
        print('min', df[col].min(), 'max', df[col].max(), 'unique', df[col].nunique())
