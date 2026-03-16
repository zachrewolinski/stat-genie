import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')

expected_missing = np.exp(df['genus']) * df['age']

sum_expected = expected_missing.groupby(df['prob_male']).sum()
num_amtl = df.groupby('prob_male')['num_amtl'].first()

print('Correlation sum_exp*age vs num_amtl', sum_expected.corr(num_amtl))
print('Mean diff', (sum_expected - num_amtl).mean(), 'std', (sum_expected - num_amtl).std())
print(pd.DataFrame({'sum_expected':sum_expected.head(), 'num_amtl':num_amtl.head(), 'diff':(sum_expected-num_amtl).head()}))
