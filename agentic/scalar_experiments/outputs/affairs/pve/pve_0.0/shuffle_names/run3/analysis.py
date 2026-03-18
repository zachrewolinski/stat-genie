import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('affairs.csv')
print(df.head())
print('\nSummary:')
print(df.describe(include='all'))

# Show unique values for each column (up to 20)
for col in df.columns:
    uniq = pd.unique(df[col])
    print(f"\n{col} unique sample ({len(uniq)}): {sorted(uniq)[:20]}")
