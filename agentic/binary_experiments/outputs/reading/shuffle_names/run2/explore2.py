import pandas as pd
import numpy as np

df = pd.read_csv('reading.csv')
print(df.isna().mean().sort_values())

print('\nCounts for dyslexia_bin:', df['dyslexia_bin'].value_counts(dropna=False))
print('Counts for correct_rate:', df['correct_rate'].value_counts(dropna=False))
