import pandas as pd


df = pd.read_csv('amtl.csv')

# compute per specimen sum of age across 3 classes
sum_age = df.groupby('prob_male')['age'].sum()
# compare to num_amtl (constant per specimen)
num_amtl = df.groupby('prob_male')['num_amtl'].first()

print('Correlation sum(age) vs num_amtl', sum_age.corr(num_amtl))
print('Correlation sum(age) vs pop', sum_age.corr(df.groupby('prob_male')['pop'].first()))
print('Correlation sum(age) vs stdev_age', sum_age.corr(df.groupby('prob_male')['stdev_age'].first()))

# check if num_amtl approx sum(age)
print('mean difference num_amtl - sum(age):', (num_amtl - sum_age).mean())
print('min/max difference', (num_amtl - sum_age).min(), (num_amtl - sum_age).max())

# compute per specimen sum of genus across classes, see correlation with num_amtl
sum_genus = df.groupby('prob_male')['genus'].sum()
print('Correlation sum(genus) vs num_amtl', sum_genus.corr(num_amtl))
print('Correlation sum(genus) vs pop', sum_genus.corr(df.groupby('prob_male')['pop'].first()))

