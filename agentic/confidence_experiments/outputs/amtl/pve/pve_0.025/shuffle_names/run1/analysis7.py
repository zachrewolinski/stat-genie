import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')

# Stats by sockets (tooth class)
print(df.groupby('sockets')[['genus','age']].agg(['mean','std','min','max']))

# Check if age counts by sockets correspond to typical counts (anterior, posterior, premolar)
print('\nAge counts by sockets:')
print(df.groupby('sockets')['age'].value_counts().head(20))

