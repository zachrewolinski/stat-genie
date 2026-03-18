import pandas as pd


df = pd.read_csv('affairs.csv')
print(df.head())
print('\nUnique values per column (sorted, first 15):')
for col in df.columns:
    uniq = sorted(df[col].dropna().unique())
    print(col, len(uniq), uniq[:15])

