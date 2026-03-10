import pandas as pd
import numpy as np

df = pd.read_csv('reading.csv')
# compute wpm from feature5 (time minus scrolling)
wpm5 = df['feature7'] / (df['feature5'] / 60000.0)
wpm4 = df['feature7'] / (df['feature4'] / 60000.0)

for name, wpm in [('wpm4', wpm4), ('wpm5', wpm5)]:
    corr = np.corrcoef(df['feature20'], wpm)[0,1]
    print(name, 'corr', corr)
    # check ratio of feature20 to wpm
    ratio = df['feature20'] / wpm
    print(name, 'ratio median', np.nanmedian(ratio), 'mean', np.nanmean(ratio))

# check if feature20 equals wpm4 or wpm5 for some rows
print('feature20 stats', df['feature20'].describe())
print('wpm4 stats', wpm4.describe())
print('wpm5 stats', wpm5.describe())

# check sample differences
print(df[['feature4','feature5','feature7','feature20']].head())
print('abs diff wpm4', np.nanmedian(np.abs(df['feature20']-wpm4)))
print('abs diff wpm5', np.nanmedian(np.abs(df['feature20']-wpm5)))
