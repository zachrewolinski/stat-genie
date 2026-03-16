import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')

age = df['pop']  # likely age at death

for col in ['genus','num_amtl','age','stdev_age']:
    corr = np.corrcoef(age, df[col])[0,1]
    print('corr(pop, %s):' % col, corr)

# also check corr between num_amtl and age (sockets count)
print('corr(num_amtl, age):', np.corrcoef(df['num_amtl'], df['age'])[0,1])
print('corr(genus, age):', np.corrcoef(df['genus'], df['age'])[0,1])
