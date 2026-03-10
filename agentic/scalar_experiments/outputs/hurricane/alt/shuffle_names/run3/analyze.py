import pandas as pd
import numpy as np

df = pd.read_csv('hurricane.csv')
print(df.head())
print(df.describe(include='all'))

# show unique ranges
for col in df.columns:
    print('\n', col)
    print(df[col].dtype)
    print(df[col].head(5).tolist())
    if df[col].dtype!='object':
        print('min', df[col].min(), 'max', df[col].max(), 'mean', df[col].mean())
    else:
        print('unique', df[col].nunique())
