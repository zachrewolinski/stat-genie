import pandas as pd


df=pd.read_csv('reading.csv')
mask=~df['img_width'].isna() & ~df['dyslexia_bin'].isna()
print(pd.crosstab(df.loc[mask,'img_width'], df.loc[mask,'dyslexia_bin']))

mask=~df['img_width'].isna() & ~df['correct_rate'].isna()
print('\nimg_width vs correct_rate')
print(pd.crosstab(df.loc[mask,'img_width'], df.loc[mask,'correct_rate']))

