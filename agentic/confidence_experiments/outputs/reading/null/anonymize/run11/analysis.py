import pandas as pd
import numpy as np

path = 'reading.csv'

df = pd.read_csv(path)
print('shape', df.shape)
print(df.head())

# basic stats for candidate variables
cols = df.columns.tolist()
print('columns', cols)

# compute derived reading speed using feature7 words and feature5 (reading time without scroll) and feature4 (total time)
# feature4/5 are in ms. speed wpm = words / (time_min)
for time_col in ['feature4','feature5']:
    speed = df['feature7'] / (df[time_col] / 60000.0)
    print(time_col, speed.describe())

# describe feature20
print('feature20 desc', df['feature20'].describe())

# correlation between feature20 and derived speed for a sample
speed5 = df['feature7'] / (df['feature5'] / 60000.0)
speed4 = df['feature7'] / (df['feature4'] / 60000.0)
print('corr feature20 vs speed5', df['feature20'].corr(speed5))
print('corr feature20 vs speed4', df['feature20'].corr(speed4))

# check if feature20 is close to speed5 or speed4
print('feature20 - speed5', (df['feature20'] - speed5).describe())
print('feature20 - speed4', (df['feature20'] - speed4).describe())

# quick check: any nonpositive times or words
print('nonpositive time4', (df['feature4']<=0).sum())
print('nonpositive time5', (df['feature5']<=0).sum())
print('nonpositive words', (df['feature7']<=0).sum())

