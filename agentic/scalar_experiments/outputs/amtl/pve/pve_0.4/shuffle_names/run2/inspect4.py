import pandas as pd

_df = pd.read_csv('amtl.csv')

# Sum of genus per specimen
sum_genus = _df.groupby('prob_male')['genus'].sum()
# Compare to num_amtl (constant per specimen)
num_amtl = _df.groupby('prob_male')['num_amtl'].first()

corr = sum_genus.corr(num_amtl)
print('corr(sum_genus, num_amtl):', corr)

# Also check if exp(genus) or genus untransformed relate
import numpy as np
corr_exp = np.exp(_df.groupby('prob_male')['genus'].sum()).corr(num_amtl)
print('corr(exp(sum_genus), num_amtl):', corr_exp)

# If genus maybe proportion (logit), inverse logit
from scipy.special import expit
sum_prop = _df.groupby('prob_male')['genus'].apply(lambda x: expit(x).sum())
print('corr(sum_invlogit(genus), num_amtl):', sum_prop.corr(num_amtl))

# Check if genus maybe log(count+1): exp(genus)-1
sum_count_est = _df.groupby('prob_male')['genus'].apply(lambda x: (np.exp(x)-1).sum())
print('corr(sum_expminus1(genus), num_amtl):', sum_count_est.corr(num_amtl))

# Check if genus itself proportional to num_amtl by scaling
ratio = (sum_genus / num_amtl).describe()
print('ratio sum_genus/num_amtl:', ratio)

