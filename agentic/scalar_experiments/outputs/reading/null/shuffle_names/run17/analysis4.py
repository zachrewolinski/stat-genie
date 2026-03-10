import pandas as pd

df = pd.read_csv('reading.csv')

for col in ['device','dyslexia','dyslexia_bin','correct_rate']:
    print(col, df[col].value_counts(dropna=False).head(10))
