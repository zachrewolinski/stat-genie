import pandas as pd

df = pd.read_csv('reading.csv')
print('dyslexia_bin unique:', df['dyslexia_bin'].dropna().unique())
print('dyslexia unique:', df['dyslexia'].dropna().unique())
print('language unique:', df['language'].dropna().unique())
print('img_width unique:', df['img_width'].dropna().unique())
