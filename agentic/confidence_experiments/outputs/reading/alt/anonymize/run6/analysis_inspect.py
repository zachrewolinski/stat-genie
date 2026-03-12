import pandas as pd
import numpy as np

path = 'reading.csv'
df = pd.read_csv(path)
print(df.head())
print(df.dtypes)
print('shape', df.shape)
print(df.describe(include='all').T.head())

for time_col in ['feature4','feature5']:
    wpm = df['feature7'] / (df[time_col]/60000)
    corr = np.corrcoef(wpm, df['feature20'])[0,1]
    print('corr feature20 with wpm using', time_col, corr)
    print('wpm stats', wpm.describe())
print('feature20 stats', df['feature20'].describe())

print('corr feature20 with feature4', df['feature20'].corr(df['feature4']))
print('corr feature20 with feature5', df['feature20'].corr(df['feature5']))

print('mean wpm f5', (df['feature7']/(df['feature5']/60000)).mean())

print(df['feature17'].value_counts(dropna=False).head())
print(df['feature12'].value_counts(dropna=False).head())
