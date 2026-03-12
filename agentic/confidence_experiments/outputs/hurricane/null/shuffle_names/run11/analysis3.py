import pandas as pd
import numpy as np
import statsmodels.api as sm

path = 'hurricane.csv'
df = pd.read_csv(path)
rename = {
    'ndam': 'id',
    'wind': 'year',
    'alldeaths': 'storm_name',
    'category': 'feminity_rating',
    'ndam15': 'min_pressure',
    'masfem_mturk': 'female_binary',
    'gender_mf': 'ss_category',
    'name': 'deaths',
    'elapsedyrs': 'damage_2013',
    'masfem': 'years_elapsed',
    'min': 'data_source',
    'ind': 'mturk_rating',
    'year': 'max_wind',
    'source': 'damage_2015',
}

df = df.rename(columns=rename)

for col in ['deaths', 'damage_2013', 'damage_2015']:
    df[f'log_{col}'] = np.log1p(df[col])

controls = ['ss_category', 'max_wind', 'min_pressure', 'year']


def fit(y, xcols):
    X = sm.add_constant(df[xcols])
    model = sm.OLS(y, X, missing='drop').fit()
    return model

models = {
    'log_deaths ~ mturk_rating + controls': fit(df['log_deaths'], ['mturk_rating'] + controls),
    'log_damage_2015 ~ feminity_rating + controls': fit(df['log_damage_2015'], ['feminity_rating'] + controls),
    'log_damage_2015 ~ female_binary + controls': fit(df['log_damage_2015'], ['female_binary'] + controls),
    'log_damage_2015 ~ mturk_rating + controls': fit(df['log_damage_2015'], ['mturk_rating'] + controls),
}

for label, model in models.items():
    print('\n', label)
    print(model.summary().tables[1])

# output key p-values
for label, model in models.items():
    for term in ['feminity_rating','female_binary','mturk_rating']:
        if term in model.params:
            print(f"{label} | {term} coef={model.params[term]:.4f} p={model.pvalues[term]:.4f}")
