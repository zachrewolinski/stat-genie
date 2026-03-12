import pandas as pd
import numpy as np
from scipy.special import expit
amtl = pd.read_csv('amtl.csv')
prop = expit(amtl['genus'])
calc = prop * amtl['age']
print('calc min max', calc.min(), calc.max())
print('corr calc vs ???')
print('corr with num_amtl', np.corrcoef(calc, amtl['num_amtl'])[0,1])
print('corr with pop', np.corrcoef(calc, amtl['pop'])[0,1])
