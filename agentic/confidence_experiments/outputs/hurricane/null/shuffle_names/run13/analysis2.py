import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import NegativeBinomial, Poisson

_df = pd.read_csv('hurricane.csv')

# Map columns based on info.json descriptions
fem_rating = 'category'
fem_binary = 'masfem_mturk'
fem_mturk = 'ind'
outcome = 'name'
controls = ['gender_mf', 'year', 'ndam15']

# Prepare
model_cols = [outcome, fem_rating, fem_binary, fem_mturk] + controls
for col in model_cols:
    _df[col] = pd.to_numeric(_df[col], errors='coerce')

model_df = _df[model_cols].dropna().copy()

# Fit NB2 (discrete) for each femininity measure

def fit_nb(y, X, label):
    X = sm.add_constant(X)
    model = NegativeBinomial(y, X)
    res = model.fit(disp=False)
    print('\n', label)
    print(res.summary())
    return res

# Fit Poisson with robust SE (HC0)

def fit_poisson(y, X, label):
    X = sm.add_constant(X)
    model = Poisson(y, X)
    res = model.fit(disp=False)
    res_robust = res.get_robustcov_results(cov_type='HC0')
    print('\n', label)
    print(res_robust.summary())
    return res_robust

for var, label in [(fem_rating, 'nb2_fem_rating_controls'), (fem_binary, 'nb2_fem_binary_controls'), (fem_mturk, 'nb2_fem_mturk_controls')]:
    fit_nb(model_df[outcome], model_df[[var] + controls], label)

for var, label in [(fem_rating, 'pois_fem_rating_controls'), (fem_binary, 'pois_fem_binary_controls'), (fem_mturk, 'pois_fem_mturk_controls')]:
    fit_poisson(model_df[outcome], model_df[[var] + controls], label)

