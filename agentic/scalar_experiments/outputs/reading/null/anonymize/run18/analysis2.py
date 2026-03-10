import pandas as pd
import numpy as np

df = pd.read_csv('reading.csv')
# compute derived reading speed from feature7 words and feature5 (reading time minus scrolling) in ms
speed = df['feature7'] / (df['feature5'] / 60000.0)
# also from feature4 total time
speed_total = df['feature7'] / (df['feature4'] / 60000.0)
print('speed derived from feature5 stats:', speed.describe())
print('speed derived from feature4 stats:', speed_total.describe())

# correlate with feature20
corr1 = speed.corr(df['feature20'])
corr2 = speed_total.corr(df['feature20'])
print('corr feature20 with speed (feature5):', corr1)
print('corr feature20 with speed (feature4):', corr2)

# ratio feature20 / speed
ratio = df['feature20'] / speed
print('ratio stats', ratio.describe())

# check a few rows
print(df[['feature7','feature5','feature4','feature20']].head())
print(speed.head())
print(speed_total.head())
