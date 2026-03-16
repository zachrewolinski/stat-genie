import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')

# check if num_amtl <= age (if age is sockets)
viol = (df['num_amtl'] > df['age']).mean()
print('fraction num_amtl > age:', viol)

# check if age <= num_amtl (if num_amtl is sockets)
viol2 = (df['age'] > df['num_amtl']).mean()
print('fraction age > num_amtl:', viol2)

# check if num_amtl <= pop (if pop is sockets?)
print('fraction num_amtl > pop:', (df['num_amtl'] > df['pop']).mean())

# check if age <= pop (age at death larger than sockets count) generally
print('fraction age > pop:', (df['age'] > df['pop']).mean())

