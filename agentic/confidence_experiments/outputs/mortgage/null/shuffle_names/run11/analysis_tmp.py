import pandas as pd
import numpy as np

df = pd.read_csv('mortgage.csv')
print(df.head())
print(df.describe(include='all'))
print(df.dtypes)
print(df.isna().sum())

# check unique values for key binary columns
for col in df.columns:
    uniq = pd.unique(df[col])
    if len(uniq) <= 5:
        print(col, sorted(uniq))

# compute correlation between female and other columns
if 'female' in df.columns:
    for col in df.columns:
        if col != 'female':
            if df[col].nunique() <= 2:
                # compute difference in mean for female vs male
                means = df.groupby('female')[col].mean()
                print('female vs', col, means.to_dict())

# cross-tab of female with deny and accept if exist
for target in ['deny', 'accept']:
    if target in df.columns:
        ct = pd.crosstab(df['female'], df[target])
        print('crosstab female vs', target)
        print(ct)

