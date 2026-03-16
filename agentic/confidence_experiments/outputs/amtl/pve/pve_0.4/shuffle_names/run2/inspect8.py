import pandas as pd
import numpy as np

_df = pd.read_csv('amtl.csv')

sum_age = _df.groupby('prob_male')['age'].sum()
# compare to num_amtl and maybe pop
num_amtl = _df.groupby('prob_male')['num_amtl'].first()
print('corr(sum(age), num_amtl):', sum_age.corr(num_amtl))
print('sum(age) describe', sum_age.describe())

# compare to pop (age)
pop = _df.groupby('prob_male')['pop'].first()
print('corr(sum(age), pop):', sum_age.corr(pop))

# check if sum age approx constant maybe 30? typical total sockets count? maybe 28 or 32
print('sum(age) unique sample:', sorted(sum_age.unique())[:10])
