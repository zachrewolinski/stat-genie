import pandas as pd

df = pd.read_csv('amtl.csv')

print('age > genus count', (df['age'] > df['genus']).sum())
print('age <= genus count', (df['age'] <= df['genus']).sum())

# if age is missing count and genus is sockets, then missing<=sockets? check
print('age>genus (missing>sockets)', (df['age'] > df['genus']).mean())

# check if age <= genus majority? no

# check if genus <= num_amtl? etc
for denom in ['age','pop','num_amtl']:
    print('missing=age denom', denom, 'violations', (df['age']>df[denom]).sum())
