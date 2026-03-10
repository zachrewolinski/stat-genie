import pandas as pd

df = pd.read_csv('reading.csv')

for col in ['dyslexia','device','dyslexia_bin']:
    print('\n', col)
    print(df.groupby(col)['running_time'].describe())
