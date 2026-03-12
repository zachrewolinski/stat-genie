import pandas as pd


df=pd.read_csv('reading.csv')
for col in ['language','device','dyslexia','dyslexia_bin','correct_rate']:
    print('\n',col)
    print(df[col].value_counts(dropna=False).head(10))
