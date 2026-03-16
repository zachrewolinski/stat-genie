import pandas as pd
import numpy as np

# Load data
amtl = pd.read_csv('amtl.csv')
print(amtl.head())
print(amtl.describe(include='all'))
print(amtl.dtypes)

# Check num_amtl summary
print('num_amtl min/max', amtl['num_amtl'].min(), amtl['num_amtl'].max())
print('num_amtl unique sample', amtl['num_amtl'].head())

# check sockets
print('sockets min/max', amtl['sockets'].min(), amtl['sockets'].max())

# Check genus counts
print(amtl['genus'].value_counts())

# Check if num_amtl approx integer counts
print('num_amtl integer? proportion', np.mean(np.isclose(amtl['num_amtl'], np.round(amtl['num_amtl']))))

