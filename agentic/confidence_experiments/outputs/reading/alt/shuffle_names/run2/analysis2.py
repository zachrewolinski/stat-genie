import pandas as pd
import numpy as np

pd.set_option('display.width', 200)

df = pd.read_csv('reading.csv')

# Basic summary
print('rows', len(df))

# correlations
for col in ['adjusted_running_time','age','gender','running_time']:
    if col in df.columns:
        print(col, 'min', df[col].min(), 'max', df[col].max(), 'median', df[col].median())

# check relation between adjusted_running_time and age and gender
print('check adjusted - age - gender')
print(((df['adjusted_running_time'] - df['age']) - df['gender']).describe())

# check ratio between adjusted_running_time and running_time
print('adjusted/running summary')
ratio = df['adjusted_running_time'] / df['running_time']
print(ratio.describe())

# check running_time vs num_words correlation
print('corr num_words vs running_time', df['num_words'].corr(df['running_time']))
print('corr num_words vs adjusted', df['num_words'].corr(df['adjusted_running_time']))
print('corr num_words vs age', df['num_words'].corr(df['age']))

# compute words per second using running_time interpreted as seconds
wps = df['num_words'] / df['running_time']
print('wps summary', wps.describe())

# words per second using adjusted_running_time (ms)
wps2 = df['num_words'] / (df['adjusted_running_time']/1000.0)
print('wps2 summary', wps2.describe())

# check if adjusted_running_time maybe in microseconds? if divide by 1000 and 1000 etc
wps3 = df['num_words'] / (df['adjusted_running_time']/100.0)
print('wps3 summary', wps3.describe())

# check wpm with running_time as seconds
wpm = wps*60
print('wpm summary', wpm.describe())

# identify if running_time ~ age/100? etc
for factor in [1,10,100,1000]:
    diff = (df['age']/factor - df['running_time'])
    print('age/',factor,'diff median',diff.median(),'std',diff.std())

# check unique counts for language (0/1) by reader_view etc
print(df.groupby('language')['reader_view'].value_counts().head(10))

# Dyslexia group counts
print('dyslexia counts', df['dyslexia'].value_counts(dropna=False))

# show distribution of running_time in dyslexia vs not
print(df.groupby('dyslexia')['running_time'].describe())
