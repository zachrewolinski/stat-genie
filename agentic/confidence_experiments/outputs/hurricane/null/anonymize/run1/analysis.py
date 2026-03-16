import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = 'hurricane.csv'
df = pd.read_csv(path)

# Rename columns for clarity
rename = {
    'feature1': 'id',
    'feature2': 'year',
    'feature3': 'name',
    'feature4': 'feminity_index',
    'feature5': 'min_pressure',
    'feature6': 'female_binary',
    'feature7': 'category',
    'feature8': 'deaths',
    'feature9': 'damage_2013',
    'feature10': 'years_since',
    'feature11': 'source',
    'feature12': 'feminity_mturk',
    'feature13': 'max_wind',
    'feature14': 'damage_2015',
}

df = df.rename(columns=rename)

# Outcomes

df['log_deaths'] = np.log1p(df['deaths'])

# Controls for storm severity and time
controls = ['min_pressure', 'max_wind', 'category', 'year']

# OLS on log deaths
formula_ols = 'log_deaths ~ feminity_index + ' + ' + '.join(controls)
ols_model = smf.ols(formula_ols, data=df).fit()

# Alternative femininity measure
formula_ols_mturk = 'log_deaths ~ feminity_mturk + ' + ' + '.join(controls)
ols_model_mturk = smf.ols(formula_ols_mturk, data=df).fit()

# Binary gender
formula_ols_binary = 'log_deaths ~ female_binary + ' + ' + '.join(controls)
ols_model_binary = smf.ols(formula_ols_binary, data=df).fit()

# Negative binomial on deaths
formula_nb = 'deaths ~ feminity_index + ' + ' + '.join(controls)
nb_model = smf.glm(formula_nb, data=df, family=sm.families.NegativeBinomial()).fit()

# Correlations
corr_raw = df['feminity_index'].corr(df['deaths'])
corr_log = df['feminity_index'].corr(df['log_deaths'])

results = {
    'n': int(len(df)),
    'corr_raw': float(corr_raw),
    'corr_log': float(corr_log),
    'ols': {
        'coef': float(ols_model.params['feminity_index']),
        'pvalue': float(ols_model.pvalues['feminity_index']),
        'r2': float(ols_model.rsquared),
    },
    'ols_mturk': {
        'coef': float(ols_model_mturk.params['feminity_mturk']),
        'pvalue': float(ols_model_mturk.pvalues['feminity_mturk']),
        'r2': float(ols_model_mturk.rsquared),
    },
    'ols_binary': {
        'coef': float(ols_model_binary.params['female_binary']),
        'pvalue': float(ols_model_binary.pvalues['female_binary']),
        'r2': float(ols_model_binary.rsquared),
    },
    'nb': {
        'coef': float(nb_model.params['feminity_index']),
        'pvalue': float(nb_model.pvalues['feminity_index']),
    }
}

print(json.dumps(results, indent=2))
