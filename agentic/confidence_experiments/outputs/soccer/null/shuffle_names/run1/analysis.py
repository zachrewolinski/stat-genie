import pandas as pd
import numpy as np

path = 'soccer.csv'

df = pd.read_csv(path)
print(df.shape)
print(df.head())
print(df.columns.tolist())
print(df.dtypes)

# basic missingness
print(df.isna().mean().sort_values(ascending=False).head(10))

# try identify skin tone variables: likely rater1 and rater2? but see columns

