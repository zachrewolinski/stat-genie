import pandas as pd
import numpy as np

pd.set_option('display.width', 200)

path = 'hurricane.csv'
df = pd.read_csv(path)

# Check relationship between wind (year) and masfem
print('wind range', df['wind'].min(), df['wind'].max())
print('masfem range', df['masfem'].min(), df['masfem'].max())

# try see if masfem equals wind-1949
print('wind-1949 unique?', (df['wind']-1949).describe())
print('correlation wind vs masfem', df['wind'].corr(df['masfem']))

# check if name is deaths - look at max etc
print('name range', df['name'].min(), df['name'].max())

# check masfem_mturk vs alldeaths? maybe female indicator
print('masfem_mturk value counts', df['masfem_mturk'].value_counts())

# See summary of deaths by female indicator
print(df.groupby('masfem_mturk')['name'].describe())

# correlation femininity index (category?) and deaths
print('corr category vs deaths', df['category'].corr(df['name']))
print('corr ind vs deaths', df['ind'].corr(df['name']))

