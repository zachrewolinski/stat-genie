import pandas as pd

df = pd.read_csv('reading.csv')

# compare dyslexia_bin with dyslexia and device columns
for col in ['dyslexia','device']:
    tmp = df[[col,'dyslexia_bin']].dropna()
    ctab = tmp.groupby(col)['dyslexia_bin'].mean()
    print('\n', col)
    print(ctab)
