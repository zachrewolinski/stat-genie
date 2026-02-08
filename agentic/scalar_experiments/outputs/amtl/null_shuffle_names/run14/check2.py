import pandas as pd

df = pd.read_csv('amtl.csv')

print('corr pop-num_amtl', df['pop'].corr(df['num_amtl']))
print('corr pop-stdev_age', df['pop'].corr(df['stdev_age']))
print('corr num_amtl-stdev_age', df['num_amtl'].corr(df['stdev_age']))

# examine num_amtl by tooth_class (genus)
print(df.groupby('tooth_class')['num_amtl'].describe())

print(df.groupby('tooth_class')['pop'].describe())

# stdev_age distribution
print(df['stdev_age'].value_counts())
