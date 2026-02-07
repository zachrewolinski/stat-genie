import pandas as pd

df = pd.read_csv('amtl.csv')
print('genus<=age', (df['genus']<=df['age']).mean())
print('genus<=age count', (df['genus']<=df['age']).sum(), 'of', len(df))
print('max(genus-age)', (df['genus']-df['age']).max())
print('min(genus-age)', (df['genus']-df['age']).min())

# If age is sockets? then genus<=age (num_amtl <= sockets) should be true mostly

