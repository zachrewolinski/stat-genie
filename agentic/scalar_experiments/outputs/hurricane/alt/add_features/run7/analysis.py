import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Load data
_df = pd.read_csv('hurricane.csv')

# Basic cleaning
# Keep relevant columns and drop rows with missing values in these columns
cols = [
    'alldeaths', 'masfem', 'gender_mf', 'category', 'wind', 'min', 'ndam15', 'year'
]

_df = _df[cols].dropna().copy()

# Transformations to handle skew
_df['log_deaths'] = np.log1p(_df['alldeaths'])
_df['log_ndam15'] = np.log1p(_df['ndam15'])

# Model 1: Femininity index (continuous)
model1 = smf.ols(
    formula='log_deaths ~ masfem + category + wind + min + log_ndam15 + year',
    data=_df
).fit(cov_type='HC3')

# Model 2: Binary gender indicator
model2 = smf.ols(
    formula='log_deaths ~ gender_mf + category + wind + min + log_ndam15 + year',
    data=_df
).fit(cov_type='HC3')

# Simple correlation (Spearman) between masfem and deaths
spearman_corr = _df[['masfem', 'alldeaths']].corr(method='spearman').iloc[0, 1]

results = {
    'n': int(_df.shape[0]),
    'spearman_masfem_alldeaths': float(spearman_corr),
    'model1': {
        'coef_masfem': float(model1.params['masfem']),
        'p_masfem': float(model1.pvalues['masfem']),
        'r2': float(model1.rsquared),
        'adj_r2': float(model1.rsquared_adj),
    },
    'model2': {
        'coef_gender_mf': float(model2.params['gender_mf']),
        'p_gender_mf': float(model2.pvalues['gender_mf']),
        'r2': float(model2.rsquared),
        'adj_r2': float(model2.rsquared_adj),
    }
}

print(results)
