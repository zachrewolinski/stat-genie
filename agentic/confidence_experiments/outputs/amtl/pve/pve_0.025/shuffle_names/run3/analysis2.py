import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')

num_cols = ['genus','age','pop','num_amtl','stdev_age']

print('Integer-likeness (within 0.2 of nearest int):')
for c in num_cols:
    vals = df[c].values
    frac_int = np.mean(np.abs(vals - np.round(vals)) < 0.2)
    print(c, frac_int)

print('\nNon-negative fraction:')
for c in num_cols:
    print(c, np.mean(df[c] >= 0))

# Check proportion of times one column <= another (candidate missing<=sockets)
cols = num_cols
print('\nPairwise <= fractions:')
for a in cols:
    for b in cols:
        if a==b: continue
        frac = np.mean(df[a] <= df[b])
        if frac > 0.7:
            print(f'{a} <= {b}: {frac:.2f}')

print('\nCorrelation with pop (possible age at death) and num_amtl:')
for c in num_cols:
    for d in num_cols:
        if c>=d: continue
    
print(df[num_cols].corr())

# Means by genus category (tooth_class column) for numeric columns
for c in ['genus','age','pop','num_amtl','stdev_age']:
    print('\nMean', c, 'by genus(tooth_class):')
    print(df.groupby('tooth_class')[c].mean().sort_values())
