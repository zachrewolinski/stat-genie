import pandas as pd
import numpy as np

df = pd.read_csv('amtl.csv')

spec_col = 'prob_male'  # appears to be specimen ID
numeric_cols = ['genus', 'age', 'pop', 'num_amtl', 'stdev_age']

print('Rows per specimen (value counts top 5):')
print(df[spec_col].value_counts().head())

for col in numeric_cols:
    uniq_counts = df.groupby(spec_col)[col].nunique()
    print(f"\n{col} unique values per specimen: min {uniq_counts.min()}, max {uniq_counts.max()}, mean {uniq_counts.mean():.2f}")
    print('Counts distribution (value counts top):')
    print(uniq_counts.value_counts().head())

# Display a few specimens
for specimen in df[spec_col].unique()[:3]:
    sub = df[df[spec_col] == specimen]
    print(f"\nSpecimen {specimen}")
    print(sub[['sockets'] + numeric_cols])
