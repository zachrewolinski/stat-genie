import pandas as pd
import numpy as np

path = 'reading.csv'

df = pd.read_csv(path)
print(df.head())
print(df.describe(include='all').T.head(30))
print('columns', df.columns)
print('dtypes', df.dtypes)
print('nulls', df.isna().sum().sort_values(ascending=False).head(10))

# try to identify reading speed: maybe feature20
# compute correlation between feature20 and time/words
for col in ['feature4','feature5','feature6','feature7','feature8','feature9','feature10','feature19','feature20']:
    if col in df.columns:
        print(col, df[col].min(), df[col].max(), df[col].mean())

# compute wpm if possible
# if feature20 maybe reading speed; compute derived: words / reading time (feature5?)
# feature5 is time on page minus scrolling duration (ms). reading speed could be words per minute.

df['derived_wpm'] = df['feature7'] / (df['feature5'] / 60000.0)
print('derived_wpm summary', df['derived_wpm'].describe())

# compare derived_wpm with feature20 correlation
if 'feature20' in df.columns:
    corr = df[['feature20','derived_wpm']].corr().iloc[0,1]
    print('corr feature20 vs derived_wpm', corr)

# dyslexia indicator: feature17 (1) or feature12? use feature17 for binary
# reader view: feature3

# check group means for dyslexia
for dyscol in ['feature17','feature12']:
    if dyscol in df.columns:
        print('group means for', dyscol)
        print(df.groupby([dyscol, 'feature3'])['derived_wpm'].mean().unstack())

