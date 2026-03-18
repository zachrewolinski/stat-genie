import pandas as pd
import numpy as np

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 50)

df = pd.read_csv('affairs.csv')

print('columns', df.columns.tolist())

# Candidate variables for affairs count
candidates = ['education', 'age']

# Map known variables by description
# children indicator is religiousness column (yes/no)

# compute correlations with marriage rating (column 'affairs')
for col in candidates:
    # use spearman to handle non-normal
    spearman = df[[col, 'affairs']].corr(method='spearman').iloc[0,1]
    pearson = df[[col, 'affairs']].corr(method='pearson').iloc[0,1]
    print(col, 'spearman vs marriage rating (affairs col):', spearman, 'pearson', pearson)

# show distributions
for col in candidates:
    print('\n', col, df[col].describe())
    print('min', df[col].min(), 'max', df[col].max())

# check if education looks like scaled affairs counts
print('\nEducation /1000 summary')
print((df['education']/1000).describe())

# how many negatives in age
print('age negatives count', (df['age']<0).sum())

# check unique rounded values for age maybe? to nearest integer
print('age rounded counts', df['age'].round().value_counts().head())

# also check correlation with children indicator (religiousness yes/no)
child = (df['religiousness'] == 'yes').astype(int)
for col in candidates:
    spearman = pd.concat([df[col], child], axis=1).corr(method='spearman').iloc[0,1]
    print(col, 'spearman vs children indicator', spearman)

