import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.discrete.discrete_model import NegativeBinomial
from scipy import stats

# Load data

df = pd.read_csv('hurricane.csv')

# Variable mapping from info.json descriptions
fem_index = 'category'        # femininity index (1-11)
female_binary = 'masfem_mturk'  # 0 male, 1 female
mturk_index = 'ind'           # alternative femininity rating

deaths = 'name'               # total deaths
wind_speed = 'year'           # max wind speed at landfall
pressure = 'ndam15'           # min pressure
category_ss = 'gender_mf'     # Saffir-Simpson category
year_of_hurricane = 'wind'    # year occurred

cols = [fem_index, female_binary, mturk_index, deaths, wind_speed, pressure, category_ss, year_of_hurricane]
df = df[cols].dropna().copy()

# Transform

df['log_deaths'] = np.log1p(df[deaths])

# Descriptive stats by binary name gender
female_groups = df.groupby(female_binary)[deaths].agg(['count','mean','median'])

# Tests of difference
male_deaths = df.loc[df[female_binary] == 0, deaths]
female_deaths = df.loc[df[female_binary] == 1, deaths]

t_stat, t_p = stats.ttest_ind(male_deaths, female_deaths, equal_var=False, nan_policy='omit')
try:
    mw_stat, mw_p = stats.mannwhitneyu(male_deaths, female_deaths, alternative='two-sided')
except ValueError:
    mw_stat, mw_p = np.nan, np.nan

# OLS with robust SE
formula_controls = 'year + ndam15 + gender_mf + wind'

ols_fem = smf.ols(f'log_deaths ~ category + {formula_controls}', data=df).fit(cov_type='HC3')
ols_bin = smf.ols(f'log_deaths ~ masfem_mturk + {formula_controls}', data=df).fit(cov_type='HC3')
ols_mturk = smf.ols(f'log_deaths ~ ind + {formula_controls}', data=df).fit(cov_type='HC3')

# Negative binomial (discrete) with estimated alpha
X_fem = sm.add_constant(df[['category','year','ndam15','gender_mf','wind']])
X_bin = sm.add_constant(df[['masfem_mturk','year','ndam15','gender_mf','wind']])
X_mturk = sm.add_constant(df[['ind','year','ndam15','gender_mf','wind']])

y = df[deaths].astype(int)

nb_fem = NegativeBinomial(y, X_fem).fit(disp=False)
nb_bin = NegativeBinomial(y, X_bin).fit(disp=False)
nb_mturk = NegativeBinomial(y, X_mturk).fit(disp=False)

results = {
    'n_obs': int(df.shape[0]),
    'female_groups': {
        'count_male': int(female_groups.loc[0,'count']),
        'count_female': int(female_groups.loc[1,'count']),
        'mean_male': float(female_groups.loc[0,'mean']),
        'mean_female': float(female_groups.loc[1,'mean']),
        'median_male': float(female_groups.loc[0,'median']),
        'median_female': float(female_groups.loc[1,'median']),
    },
    't_test': {'t_stat': float(t_stat), 'p_value': float(t_p)},
    'mannwhitney': {'u_stat': float(mw_stat), 'p_value': float(mw_p)},
    'ols': {
        'fem_index': {'coef': float(ols_fem.params['category']), 'p_value': float(ols_fem.pvalues['category'])},
        'female_binary': {'coef': float(ols_bin.params['masfem_mturk']), 'p_value': float(ols_bin.pvalues['masfem_mturk'])},
        'mturk_index': {'coef': float(ols_mturk.params['ind']), 'p_value': float(ols_mturk.pvalues['ind'])},
    },
    'nb': {
        'fem_index': {'coef': float(nb_fem.params['category']), 'p_value': float(nb_fem.pvalues['category'])},
        'female_binary': {'coef': float(nb_bin.params['masfem_mturk']), 'p_value': float(nb_bin.pvalues['masfem_mturk'])},
        'mturk_index': {'coef': float(nb_mturk.params['ind']), 'p_value': float(nb_mturk.pvalues['ind'])},
    }
}

print(json.dumps(results, indent=2))
