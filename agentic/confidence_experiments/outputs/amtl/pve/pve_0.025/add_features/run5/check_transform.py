import pandas as pd
import numpy as np
from scipy.special import expit

df = pd.read_csv('amtl.csv')
# compute expected counts from logit transform
p = expit(df['num_amtl'])
counts = p * df['sockets']
# check closeness to integer counts
close = np.isclose(counts, np.round(counts), atol=0.05)  # 0.05 tolerance
print('fraction close to integer', close.mean())
# show some examples
print(counts.head(10))
print(np.round(counts.head(10)))
# check distribution min max
print('counts min max', counts.min(), counts.max())

# if num_amtl is centered logit? try reverse? hmm
