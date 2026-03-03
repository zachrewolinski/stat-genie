import pandas as pd

df = pd.read_csv('amtl.csv')
print(df.head())
print(df.dtypes)
print('genus counts:\n', df['genus'].value_counts(dropna=False))
print('tooth_class counts:\n', df['tooth_class'].value_counts(dropna=False))
print(df[['num_amtl','sockets','age','prob_male']].describe())
