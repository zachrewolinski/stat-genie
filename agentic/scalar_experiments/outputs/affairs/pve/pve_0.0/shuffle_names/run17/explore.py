import pandas as pd
import numpy as np

df = pd.read_csv('affairs.csv')

# Basic stats for numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns

for col in numeric_cols:
    series = df[col]
    zeros = (series == 0).sum()
    neg = (series < 0).sum()
    pos = (series > 0).sum()
    print(f"{col}: min={series.min()}, max={series.max()}, mean={series.mean():.3f}, std={series.std():.3f}, zeros={zeros}, neg={neg}, pos={pos}, nunique={series.nunique()}")

# distribution of 'age' variable
print('\nAge value counts (top 10):')
print(df['age'].value_counts().head(10))

# distribution of 'affairs' variable
print('\nAffairs value counts:')
print(df['affairs'].value_counts().sort_index())

# distribution of 'religiousness'
print('\nReligiousness value counts:')
print(df['religiousness'].value_counts())

