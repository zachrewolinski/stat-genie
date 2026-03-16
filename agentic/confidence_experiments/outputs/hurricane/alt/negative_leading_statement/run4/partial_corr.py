import pandas as pd
import numpy as np
import pingouin as pg

# Load data

df = pd.read_csv('hurricane.csv')

# log deaths

df['log_deaths'] = np.log1p(df['alldeaths'])

# Partial correlation of log_deaths and masfem controlling for wind, min, category

pcorr = pg.partial_corr(data=df, x='masfem', y='log_deaths', covar=['wind','min','category'])
print(pcorr)

# Also for masfem_mturk
pcorr2 = pg.partial_corr(data=df, x='masfem_mturk', y='log_deaths', covar=['wind','min','category'])
print(pcorr2)

# for gender_mf
pcorr3 = pg.partial_corr(data=df, x='gender_mf', y='log_deaths', covar=['wind','min','category'])
print(pcorr3)
