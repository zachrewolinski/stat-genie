import pandas as pd

df = pd.read_csv('amtl.csv')

print('age <= num_amtl proportion', (df['age'] <= df['num_amtl']).mean())
print('age <= pop proportion', (df['age'] <= df['pop']).mean())

# check if age (integer) <= num_amtl (float) always
print('violations age > num_amtl', (df['age'] > df['num_amtl']).sum())
