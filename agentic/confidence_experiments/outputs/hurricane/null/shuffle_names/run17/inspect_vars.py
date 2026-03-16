import pandas as pd
import numpy as np

path='hurricane.csv'
df=pd.read_csv(path)

# correlation with binary gender indicator
binary='masfem_mturk'
cont_cols=['category','ind','masfem']
for col in cont_cols:
    corr = df[[col,binary]].corr().iloc[0,1]
    print(col, 'corr with binary', corr)

# group means for continuous variables by binary
for col in cont_cols:
    print('\n', col)
    print(df.groupby(binary)[col].describe())

# inspect categories for gender_mf
print('\n gender_mf unique', sorted(df['gender_mf'].unique()))

# look at name (deaths) distribution
print('\n name (deaths) describe')
print(df['name'].describe())

# check if any zeros
print('deaths zero count', (df['name']==0).sum())

