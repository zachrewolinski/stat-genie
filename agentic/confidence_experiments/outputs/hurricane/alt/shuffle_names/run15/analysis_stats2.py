import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

# Load data
path = 'hurricane.csv'
df = pd.read_csv(path)

# Map columns

df['deaths'] = df['name']
df['log_deaths'] = np.log1p(df['deaths'])

df['fem_score'] = df['category']
df['fem_mturk'] = df['ind']
df['fem_binary'] = df['masfem_mturk']

df['wind_speed'] = df['year']
df['min_pressure'] = df['ndam15']
df['ss_category'] = df['gender_mf']

df['storm_year'] = df['wind']
df['damage_2015'] = df['source']

# Group stats
male = df.loc[df['fem_binary']==0,'deaths']
female = df.loc[df['fem_binary']==1,'deaths']

# t-test on log deaths
male_log = np.log1p(male)
female_log = np.log1p(female)

ttest = stats.ttest_ind(female_log, male_log, equal_var=False)

# Mann-Whitney U
mw = stats.mannwhitneyu(female, male, alternative='two-sided')

print('Female mean deaths', female.mean(), 'Male mean deaths', male.mean())
print('Female median deaths', female.median(), 'Male median deaths', male.median())
print('t-test log deaths:', ttest)
print('Mann-Whitney:', mw)

