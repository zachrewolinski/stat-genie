import pandas as pd
import numpy as np

df = pd.read_csv('affairs.csv')
print(df.head())
print('\nDescribe:')
print(df.describe(include='all'))

# unique values for categorical-ish columns
for col in df.columns:
    if df[col].dtype == 'object':
        print(col, df[col].value_counts())
    else:
        # show unique if small
        uniq = np.sort(df[col].unique())
        if len(uniq) <= 12:
            print(col, uniq)

