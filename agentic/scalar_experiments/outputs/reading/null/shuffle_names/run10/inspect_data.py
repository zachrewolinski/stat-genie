import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)

path = 'reading.csv'
df = pd.read_csv(path)
print(df.describe(include='all').transpose().head(30))

# show unique values for categorical columns
for col in df.select_dtypes(include=['object']).columns:
    print('\n', col, 'unique', df[col].nunique())
    print(df[col].value_counts().head())

# show ranges for numeric
for col in df.select_dtypes(include=[np.number]).columns:
    print('\n', col, 'min', df[col].min(), 'max', df[col].max(), 'mean', df[col].mean())
