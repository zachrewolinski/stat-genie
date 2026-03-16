import pandas as pd
import numpy as np
import statsmodels.api as sm

_df = pd.read_csv('hurricane.csv').rename(columns={
    'ndam': 'id',
    'wind': 'year',
    'alldeaths': 'name',
    'category': 'femininity_coders',
    'ndam15': 'min_pressure',
    'masfem_mturk': 'female_indicator',
    'gender_mf': 'ss_category',
    'name': 'deaths',
    'elapsedyrs': 'damage_2013',
    'masfem': 'elapsed_years',
    'min': 'source',
    'ind': 'femininity_mturk',
    'year': 'max_wind',
    'source': 'damage_2015',
})

# Negative binomial GLM for count outcome
controls = ['max_wind','min_pressure','ss_category']

for predictor in ['femininity_coders','femininity_mturk','female_indicator']:
    X = sm.add_constant(_df[[predictor] + controls])
    model = sm.GLM(_df['deaths'], X, family=sm.families.NegativeBinomial()).fit()
    print('\nNegBin:', predictor)
    print(model.summary().tables[1])
