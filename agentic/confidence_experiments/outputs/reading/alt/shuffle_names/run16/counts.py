import pandas as pd

df = pd.read_csv('reading.csv')

for col in ['device','dyslexia','dyslexia_bin','correct_rate','language']:
    print('\n', col)
    print(df[col].value_counts(dropna=False).sort_index())
