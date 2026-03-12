import pandas as pd
import numpy as np

path='reading.csv'
df=pd.read_csv(path)

# correlations with other columns
cols=['feature4','feature5','feature6','feature7','feature8','feature9','feature19','feature20']

print(df[cols].corr())

# check if feature20 equals reading speed? maybe words per minute computed from feature4 or feature5?
# compute wpm based on feature4 total time and feature5 reading time

df['wpm_total'] = df['feature7']/(df['feature4']/60000.0)
df['wpm_reading'] = df['feature7']/(df['feature5']/60000.0)

print('corr feature20 vs wpm_total', df[['feature20','wpm_total']].corr().iloc[0,1])
print('corr feature20 vs wpm_reading', df[['feature20','wpm_reading']].corr().iloc[0,1])

# check if feature20 correlates with time instead (lower = faster?)
print('corr feature20 vs feature4', df[['feature20','feature4']].corr().iloc[0,1])
print('corr feature20 vs feature5', df[['feature20','feature5']].corr().iloc[0,1])

# check typical values of feature20 by wpm quantiles
print(df['feature20'].describe())
print('wpm_total summary', df['wpm_total'].describe())
print('wpm_reading summary', df['wpm_reading'].describe())

# potential log transform of times? maybe feature20 is reading speed (wpm) scaled by ???
# check if feature20 ~ words per minute * some constant? look at ratio
ratio_total = df['feature20'] / df['wpm_total']
ratio_reading = df['feature20'] / df['wpm_reading']
print('ratio_total median', ratio_total.median(), 'ratio_reading median', ratio_reading.median())

# check if feature20 maybe words per second? wps = words/(feature5/1000)

df['wps_reading'] = df['feature7']/(df['feature5']/1000.0)
print('corr feature20 vs wps_reading', df[['feature20','wps_reading']].corr().iloc[0,1])
print('wps_reading summary', df['wps_reading'].describe())

