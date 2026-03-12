import pandas as pd

_df = pd.read_csv('amtl.csv')

# Summaries by tooth class (sockets column)
for col in ['genus','age','pop','num_amtl','stdev_age']:
    print('\nColumn', col)
    print(_df.groupby('sockets')[col].describe())

print('\nBy genus (tooth_class column) summary of age (pop):')
print(_df.groupby('tooth_class')['pop'].describe())

# Check relationships between age (pop) and num_amtl etc
print('\nCorrelations with pop (age)')
print(_df[['genus','age','pop','num_amtl','stdev_age']].corr())

# Check integer-like? counts?
for col in ['genus','age','pop','num_amtl','stdev_age']:
    frac = (_df[col] % 1 != 0).mean()
    print(col, 'fraction non-integer', frac)
