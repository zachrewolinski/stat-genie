import pandas as pd
import numpy as np


df = pd.read_csv('reading.csv')
# Derived reading speed from feature7 words and feature5 reading time minus scrolling (ms)
# wpm = words / (time_ms/60000)
# To avoid divide by zero
wpm5 = df['feature7'] / (df['feature5'] / 60000.0)
# also from feature4 total time
wpm4 = df['feature7'] / (df['feature4'] / 60000.0)

# compute correlation with feature20
corr5 = np.corrcoef(wpm5, df['feature20'])[0,1]
corr4 = np.corrcoef(wpm4, df['feature20'])[0,1]
print('corr wpm5 vs feature20', corr5)
print('corr wpm4 vs feature20', corr4)
print('wpm5 summary', wpm5.describe())
print('wpm4 summary', wpm4.describe())
print('feature20 summary', df['feature20'].describe())

# show sample rows to inspect closeness
sample = df[['feature7','feature4','feature5','feature20']].head(10)
print(sample)
# compute ratio feature20/wpm5
ratio = df['feature20'] / wpm5
print('ratio summary', ratio.describe())
