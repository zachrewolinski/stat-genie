import pandas as pd

df = pd.read_csv('amtl.csv')
print(df.groupby('tooth_class')['pop'].describe())
print('\nAge int by genus')
print(df.groupby('tooth_class')['age'].describe())
