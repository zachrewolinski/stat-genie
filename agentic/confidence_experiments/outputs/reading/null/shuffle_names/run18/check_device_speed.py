import pandas as pd

path='reading.csv'
df=pd.read_csv(path)

for col in ['device','dyslexia','correct_rate','dyslexia_bin','language']:
    print(f"\nUnique speeds per {col} value:")
    print(df.groupby(col)['speed'].nunique())
