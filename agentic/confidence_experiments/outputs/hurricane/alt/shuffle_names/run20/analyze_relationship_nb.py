import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import NegativeBinomial

# Load data
path = 'hurricane.csv'
df = pd.read_csv(path)

# Map columns
analysis_df = pd.DataFrame({
    'deaths': df['name'],
    'masfem_index': df['category'],
    'masfem_mturk_cont': df['ind'],
    'female_binary': df['masfem_mturk'],
    'hurr_category': df['gender_mf'],
    'min_pressure': df['ndam15'],
    'max_wind': df['year'],
    'year': df['wind'],
})

# Negative Binomial with estimated alpha
X = sm.add_constant(analysis_df[['masfem_index','max_wind','min_pressure','hurr_category','year']])
nb_model = NegativeBinomial(analysis_df['deaths'], X)
nb_res = nb_model.fit(disp=False)
print(nb_res.summary().as_text())

# Also NB with mturk femininity
X2 = sm.add_constant(analysis_df[['masfem_mturk_cont','max_wind','min_pressure','hurr_category','year']])
nb_model2 = NegativeBinomial(analysis_df['deaths'], X2)
nb_res2 = nb_model2.fit(disp=False)
print('\n', nb_res2.summary().as_text())

# NB with female binary
X3 = sm.add_constant(analysis_df[['female_binary','max_wind','min_pressure','hurr_category','year']])
nb_model3 = NegativeBinomial(analysis_df['deaths'], X3)
nb_res3 = nb_model3.fit(disp=False)
print('\n', nb_res3.summary().as_text())
