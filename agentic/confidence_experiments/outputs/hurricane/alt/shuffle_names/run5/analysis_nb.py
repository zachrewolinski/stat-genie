import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import NegativeBinomial

# Load data
_df = pd.read_csv('hurricane.csv')

rename_map = {
    'ndam': 'id',
    'wind': 'year',
    'alldeaths': 'hurricane_name',
    'category': 'femininity_index',
    'ndam15': 'min_pressure',
    'masfem_mturk': 'female_binary',
    'gender_mf': 'saffir_simpson_cat',
    'name': 'deaths',
    'elapsedyrs': 'damage_2013',
    'masfem': 'years_elapsed',
    'min': 'data_source',
    'ind': 'femininity_mturk',
    'year': 'max_wind_speed',
    'source': 'damage_2015',
}

df = _df.rename(columns=rename_map)

num_cols = [
    'year', 'femininity_index', 'min_pressure', 'female_binary', 'saffir_simpson_cat',
    'deaths', 'damage_2013', 'years_elapsed', 'femininity_mturk',
    'max_wind_speed', 'damage_2015'
]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors='coerce')

controls = ['max_wind_speed', 'min_pressure', 'saffir_simpson_cat', 'year']

# Negative Binomial using discrete model (estimates alpha)
results = {}
for label in ['femininity_index', 'female_binary']:
    X = df[[label] + controls]
    X = sm.add_constant(X)
    y = df['deaths']
    model = NegativeBinomial(y, X, missing='drop')
    try:
        res = model.fit(disp=False)
    except Exception as e:
        res = e
    results[label] = res

for label, res in results.items():
    print(f"\nNegative Binomial deaths ~ {label} + controls")
    if isinstance(res, Exception):
        print('Error:', res)
    else:
        print(res.summary().tables[1])
        # print alpha (overdispersion)
        if hasattr(res, 'params') and 'alpha' in res.params:
            print('alpha', res.params['alpha'])
