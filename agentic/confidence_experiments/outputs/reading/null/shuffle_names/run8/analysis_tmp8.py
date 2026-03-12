import pandas as pd

df = pd.read_csv('reading.csv')
print(df['dyslexia_bin'].value_counts(dropna=False))
