import pandas as pd
import numpy as np

df = pd.read_csv('affairs.csv')

edu = df['education']
print('unique count', edu.nunique())
# try scaling
for scale in [1,10,100,1000]:
    vals = (edu/scale).round(2)
    print('scale', scale, 'nunique', vals.nunique(), 'min', vals.min(), 'max', vals.max())

# check if education values cluster near multiples
for base in [1,5,10,25,50,100]:
    diff = ((edu/base).round() - edu/base).abs().mean()
    print('base', base, 'mean abs rounding diff', diff)

print('quantiles', edu.quantile([0,0.1,0.25,0.5,0.75,0.9,1]))
