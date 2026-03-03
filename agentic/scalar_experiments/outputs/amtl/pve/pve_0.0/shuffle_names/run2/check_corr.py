import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')

# compute correlation between genus and num_amtl
corr = np.corrcoef(df['genus'], df['num_amtl'])[0,1]
print('corr(genus, num_amtl):', corr)

# check if genus is standardized num_amtl
num = df['num_amtl']
std_num = (num - num.mean()) / num.std(ddof=0)
max_diff = np.max(np.abs(std_num - df['genus']))
print('max diff between genus and zscore(num_amtl):', max_diff)

# check if genus is standardized pop
std_pop = (df['pop'] - df['pop'].mean()) / df['pop'].std(ddof=0)
print('max diff between genus and zscore(pop):', np.max(np.abs(std_pop - df['genus'])))

# check if genus is standardized age (integer)
std_age = (df['age'] - df['age'].mean()) / df['age'].std(ddof=0)
print('max diff between genus and zscore(age):', np.max(np.abs(std_age - df['genus'])))

