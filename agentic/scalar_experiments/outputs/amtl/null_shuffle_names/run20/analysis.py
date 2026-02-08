import pandas as pd
import numpy as np

path = "amtl.csv"
df = pd.read_csv(path)
print(df.head())
print("columns", df.columns.tolist())
print(df.dtypes)

print("unique tooth_class", df['tooth_class'].unique())
print("unique sockets", df['sockets'].unique())
print("genus min max", df['genus'].min(), df['genus'].max())
print("age min max", df['age'].min(), df['age'].max())
print("pop min max", df['pop'].min(), df['pop'].max())
print("num_amtl min max", df['num_amtl'].min(), df['num_amtl'].max())
print("stdev_age unique", sorted(df['stdev_age'].unique()))

for col in ['genus','age']:
    print(col, (df[col] % 1 == 0).mean())

print(df[['genus','age','pop','num_amtl','stdev_age']].describe())

print("genus vs age correlation", df['genus'].corr(df['age']))

print("pop quantiles", df['pop'].quantile([0,0.25,0.5,0.75,1]))

print("stdev_age value counts", df['stdev_age'].value_counts().head())
