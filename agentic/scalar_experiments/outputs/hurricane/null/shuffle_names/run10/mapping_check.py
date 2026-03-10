import pandas as pd
import numpy as np

df = pd.read_csv('hurricane.csv')

# correlation matrix for numeric columns
num_cols = df.select_dtypes(include=['number']).columns
corr = df[num_cols].corr()
print(corr.loc['wind','masfem'])

# check if masfem ~ wind-1949
print(((df['wind'] - 1949) - df['masfem']).describe())

# check if year column matches wind speed (typical range 75-190)
print(df['year'].describe())

