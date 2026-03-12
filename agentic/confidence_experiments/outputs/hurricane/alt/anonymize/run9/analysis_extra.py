import pandas as pd
import numpy as np

rename = {
    'feature1': 'id',
    'feature2': 'year',
    'feature3': 'name',
    'feature4': 'femininity_index',
    'feature5': 'min_pressure',
    'feature6': 'female_binary',
    'feature7': 'category',
    'feature8': 'deaths',
    'feature9': 'damage_2013',
    'feature10': 'years_elapsed',
    'feature11': 'source',
    'feature12': 'femininity_mturk',
    'feature13': 'max_wind',
    'feature14': 'damage_2015',
}

df = pd.read_csv('hurricane.csv').rename(columns=rename)

summary = df.groupby('female_binary')['deaths'].agg(['count', 'mean', 'median', 'sum'])
print(summary)

# Mean log deaths
summary_log = df.groupby('female_binary')['deaths'].apply(lambda s: np.log1p(s).mean())
print('Mean log1p(deaths) by female_binary:', summary_log.to_dict())

# Correlation between femininity_index and log_deaths
corr = df[['femininity_index', 'deaths']].corr(method='spearman').iloc[0,1]
print('Spearman correlation:', corr)
