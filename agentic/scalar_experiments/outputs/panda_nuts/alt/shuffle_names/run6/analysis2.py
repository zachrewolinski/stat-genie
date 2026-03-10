import pandas as pd
import numpy as np

# Load and rename

df = pd.read_csv('panda_nuts.csv')
df = df.rename(columns={
    'nuts_opened': 'sex',
    'sex': 'hammer_type',
    'help': 'nuts_opened',
    'chimpanzee': 'seconds',
    'seconds': 'help'
})

df['help'] = df['help'].astype(str).str.strip().str.lower().map({'y': 1, 'n': 0})

for col in ['age', 'hammer', 'nuts_opened', 'seconds']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Efficiency

df['efficiency'] = df['nuts_opened'] / df['seconds']

# Group means
print('Efficiency by sex:')
print(df.groupby('sex')['efficiency'].agg(['count','mean','std']))

print('\nEfficiency by help:')
print(df.groupby('help')['efficiency'].agg(['count','mean','std']))

print('\nAge vs efficiency (by age):')
print(df.groupby('age')['efficiency'].mean().head())

