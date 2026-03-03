import pandas as pd
import numpy as np


df=pd.read_csv('amtl.csv')

# assume genus = logit(p) for missing rate in that class
p = 1/(1+np.exp(-df['genus']))
expected_missing = p * df['age']
# sum expected missing across classes per specimen
sum_missing = expected_missing.groupby(df['prob_male']).sum()
num_amtl = df.groupby('prob_male')['num_amtl'].first()
print('corr sum_expected_missing with num_amtl', sum_missing.corr(num_amtl))
print('summary sum_expected_missing', sum_missing.describe())
print('summary num_amtl', num_amtl.describe())

# maybe num_amtl corresponds to sum_missing + ???
print('ratio num_amtl/sum_expected_missing', (num_amtl/sum_missing).describe())
