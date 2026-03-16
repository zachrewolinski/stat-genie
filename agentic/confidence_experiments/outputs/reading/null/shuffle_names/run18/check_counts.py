import pandas as pd

path='reading.csv'
df=pd.read_csv(path)

for col in ['device','dyslexia','gender']:
    print('\n', col)
    print(df[col].value_counts(dropna=False).head(10))
