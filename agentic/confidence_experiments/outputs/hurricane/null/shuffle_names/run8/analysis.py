import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

# Load data
csv_path = 'hurricane.csv'
df = pd.read_csv(csv_path)

# Map columns based on info.json descriptions
# fem_index: continuous masculinity-femininity rating (1-11)
# female_binary: 0 male, 1 female
# mturk_index: alternative continuous rating
# deaths: total deaths
# wind_speed: max wind speed at landfall
# pressure: minimum pressure at landfall
# category_ss: Saffir-Simpson category (1-5)
# year_of_hurricane: year occurred

fem_index = 'category'
female_binary = 'masfem_mturk'
mturk_index = 'ind'
deaths = 'name'
wind_speed = 'year'
pressure = 'ndam15'
category_ss = 'gender_mf'
year_of_hurricane = 'wind'

# Clean and prepare
cols_needed = [fem_index, female_binary, mturk_index, deaths, wind_speed, pressure, category_ss, year_of_hurricane]

df = df[cols_needed].copy()

# Drop rows with missing critical values
analysis_df = df.dropna()

analysis_df['log_deaths'] = np.log1p(analysis_df[deaths])

# Descriptive stats by female_binary
female_groups = analysis_df.groupby(female_binary)[deaths].agg(['count','mean','median']).rename(index={0:'male',1:'female'})

# T-test and Mann-Whitney for deaths by binary gender
male_deaths = analysis_df.loc[analysis_df[female_binary]==0, deaths]
female_deaths = analysis_df.loc[analysis_df[female_binary]==1, deaths]

# Welch's t-test
if len(male_deaths) > 1 and len(female_deaths) > 1:
    t_stat, t_p = stats.ttest_ind(male_deaths, female_deaths, equal_var=False, nan_policy='omit')
    # Mann-Whitney (nonparametric)
    try:
        mw_stat, mw_p = stats.mannwhitneyu(male_deaths, female_deaths, alternative='two-sided')
    except ValueError:
        mw_stat, mw_p = np.nan, np.nan
else:
    t_stat, t_p, mw_stat, mw_p = np.nan, np.nan, np.nan, np.nan

# Regression models
# Model 1: log_deaths ~ fem_index + controls
formula1 = 'log_deaths ~ category + year + ndam15 + gender_mf + wind'
model1 = smf.ols(formula1, data=analysis_df).fit(cov_type='HC3')

# Model 2: log_deaths ~ female_binary + controls
formula2 = 'log_deaths ~ masfem_mturk + year + ndam15 + gender_mf + wind'
model2 = smf.ols(formula2, data=analysis_df).fit(cov_type='HC3')

# Model 3: log_deaths ~ mturk_index + controls
formula3 = 'log_deaths ~ ind + year + ndam15 + gender_mf + wind'
model3 = smf.ols(formula3, data=analysis_df).fit(cov_type='HC3')

# Negative binomial GLM for deaths (count)
# Use log link with same controls
analysis_df['deaths'] = analysis_df[deaths].astype(int)

formula_nb1 = 'deaths ~ category + year + ndam15 + gender_mf + wind'
formula_nb2 = 'deaths ~ masfem_mturk + year + ndam15 + gender_mf + wind'
formula_nb3 = 'deaths ~ ind + year + ndam15 + gender_mf + wind'

nb1 = smf.glm(formula_nb1, data=analysis_df, family=sm.families.NegativeBinomial()).fit()
nb2 = smf.glm(formula_nb2, data=analysis_df, family=sm.families.NegativeBinomial()).fit()
nb3 = smf.glm(formula_nb3, data=analysis_df, family=sm.families.NegativeBinomial()).fit()

# Collect key results
results = {
    'n_obs': int(analysis_df.shape[0]),
    'female_groups': female_groups.to_dict(),
    't_test': {'t_stat': float(t_stat), 'p_value': float(t_p)},
    'mannwhitney': {'u_stat': float(mw_stat), 'p_value': float(mw_p)},
    'ols_fem_index': {
        'coef': float(model1.params.get('category', np.nan)),
        'p_value': float(model1.pvalues.get('category', np.nan)),
        'r2': float(model1.rsquared),
    },
    'ols_female_binary': {
        'coef': float(model2.params.get('masfem_mturk', np.nan)),
        'p_value': float(model2.pvalues.get('masfem_mturk', np.nan)),
        'r2': float(model2.rsquared),
    },
    'ols_mturk_index': {
        'coef': float(model3.params.get('ind', np.nan)),
        'p_value': float(model3.pvalues.get('ind', np.nan)),
        'r2': float(model3.rsquared),
    },
    'nb_fem_index': {
        'coef': float(nb1.params.get('category', np.nan)),
        'p_value': float(nb1.pvalues.get('category', np.nan)),
    },
    'nb_female_binary': {
        'coef': float(nb2.params.get('masfem_mturk', np.nan)),
        'p_value': float(nb2.pvalues.get('masfem_mturk', np.nan)),
    },
    'nb_mturk_index': {
        'coef': float(nb3.params.get('ind', np.nan)),
        'p_value': float(nb3.pvalues.get('ind', np.nan)),
    },
}

print(json.dumps(results, indent=2))
