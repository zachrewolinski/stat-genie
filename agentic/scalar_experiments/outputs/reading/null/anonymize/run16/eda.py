import pandas as pd
import numpy as np

path = 'reading.csv'

df = pd.read_csv(path)
print(df.head())
print('columns', df.columns.tolist())
print(df.describe(include='all').T[['mean','std','min','max']].head(10))
print('feature20 stats', df['feature20'].describe())
# compute reading speed from feature7 words and feature5 time (ms)
# speed words per minute
speed = df['feature7'] / (df['feature5']/1000/60)
print('computed speed stats', speed.describe())
# check correlation between feature20 and computed speed
print('corr feature20 vs computed speed', df['feature20'].corr(speed))
# check correlation between feature20 and feature5
print('corr feature20 vs feature5', df['feature20'].corr(df['feature5']))
print('corr feature20 vs feature4', df['feature20'].corr(df['feature4']))
