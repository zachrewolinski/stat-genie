import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')

sum_exp = np.exp(df['genus']).groupby(df['prob_male']).sum()
num_amtl = df.groupby('prob_male')['num_amtl'].first()

print('Correlation sum_exp vs num_amtl', sum_exp.corr(num_amtl))
print('Mean diff', (sum_exp - num_amtl).mean(), 'std', (sum_exp - num_amtl).std())
print(pd.DataFrame({'sum_exp':sum_exp.head(), 'num_amtl':num_amtl.head(), 'diff':(sum_exp-num_amtl).head()}))
