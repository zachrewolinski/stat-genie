import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load and rename
rename = {
    'wind': 'year',
    'alldeaths': 'hurr_name',
    'category': 'masfem_coder',
    'ndam15': 'min_pressure',
    'masfem_mturk': 'female_binary',
    'gender_mf': 'ss_category',
    'name': 'deaths',
    'elapsedyrs': 'damage_2013',
    'masfem': 'years_elapsed',
    'ind': 'masfem_mturk',
    'year': 'max_wind',
    'source': 'damage_2015'
}

df = pd.read_csv('hurricane.csv').rename(columns=rename)

df['log_deaths'] = np.log1p(df['deaths'])

# Interaction with severity (max_wind)
for fem in ['masfem_coder','masfem_mturk','female_binary']:
    formula = f"log_deaths ~ {fem} * max_wind + min_pressure + ss_category + year"
    model = smf.ols(formula, data=df).fit()
    print('\n', formula)
    print(model.summary().tables[1])
