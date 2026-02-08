import pandas as pd

df = pd.read_csv('amtl.csv')
print('num_amtl unique sample', sorted(df['num_amtl'].unique())[:20])
print('num_amtl unique tail', sorted(df['num_amtl'].unique())[-10:])

# check if num_amtl corresponds to pop stdev? maybe integer? count duplicates
print('num_amtl unique count', df['num_amtl'].nunique())

# check by specimen: for a specimen, is num_amtl constant? maybe it's age stdev? by specimen id (prob_male column)
by_spec = df.groupby('prob_male')['num_amtl'].nunique()
print('num_amtl per specimen: min', by_spec.min(), 'max', by_spec.max())
print('num_amtl per specimen >1', (by_spec>1).mean())

by_spec_pop = df.groupby('prob_male')['pop'].nunique()
print('pop per specimen min', by_spec_pop.min(), 'max', by_spec_pop.max(), 'prop>1', (by_spec_pop>1).mean())

by_spec_genus = df.groupby('prob_male')['genus'].nunique()
print('genus per specimen min', by_spec_genus.min(), 'max', by_spec_genus.max(), 'prop>1', (by_spec_genus>1).mean())

by_spec_age = df.groupby('prob_male')['age'].nunique()
print('age per specimen min', by_spec_age.min(), 'max', by_spec_age.max(), 'prop>1', (by_spec_age>1).mean())

by_spec_stdev = df.groupby('prob_male')['stdev_age'].nunique()
print('stdev per specimen min', by_spec_stdev.min(), 'max', by_spec_stdev.max(), 'prop>1', (by_spec_stdev>1).mean())
