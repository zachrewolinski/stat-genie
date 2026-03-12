import pandas as pd

df = pd.read_csv('reading.csv')

missing = df.isna().mean().sort_values(ascending=False)
print(missing)
