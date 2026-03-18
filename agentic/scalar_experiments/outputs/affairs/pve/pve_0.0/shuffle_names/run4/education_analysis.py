import pandas as pd
import numpy as np

x = pd.read_csv('affairs.csv')['education']

# check if all integers
print('int?', np.all(np.equal(np.mod(x,1),0)))

# check unique differences
diffs = np.diff(np.sort(x.unique()))
print('min diff', diffs.min(), 'max diff', diffs.max())

# check divisibility by 7, 10, 100 etc
for d in [2,3,4,5,6,7,8,9,10,12,25,50,100]:
    print('divisible by', d, 'fraction', np.mean(x % d == 0))

# sample values
print('sample', x.sample(10, random_state=0).tolist())
