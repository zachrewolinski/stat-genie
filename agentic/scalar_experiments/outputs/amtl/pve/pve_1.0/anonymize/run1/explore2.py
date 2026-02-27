import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')
print(pd.crosstab(df['feature3'], df['feature8']))
print(pd.crosstab(df['feature3'], df['feature1']))
print(pd.crosstab(df['feature3'], df['feature9']).head())
# check by species and age etc
print(df.groupby('feature3')['feature5'].describe())

