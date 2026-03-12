import pandas as pd

df = pd.read_csv('reading.csv')
for col in ['device','dyslexia']:
    print('\n', col)
    print(df[col].value_counts(dropna=False))
