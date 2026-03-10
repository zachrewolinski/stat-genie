import pandas as pd
import numpy as np
import statsmodels.api as sm


df = pd.read_csv('hurricane.csv')

# Basic transforms

df['log_deaths'] = np.log1p(df['alldeaths'])
df['log_ndam15'] = np.log1p(df['ndam15'])

# Helper to run OLS with robust SE

def run_ols(data, y, X_cols, add_constant=True):
    X = data[X_cols].copy()
    if add_constant:
        X = sm.add_constant(X, has_constant='add')
    model = sm.OLS(data[y], X, missing='drop')
    res = model.fit(cov_type='HC3')
    return res

results = {}

# Model 1: log deaths ~ masfem (bivariate)
cols1 = ['masfem']
res1 = run_ols(df, 'log_deaths', cols1)
results['bivariate'] = res1

# Model 2: log deaths ~ masfem + severity controls (wind, min pressure, category)
cols2 = ['masfem', 'wind', 'min', 'category']
res2 = run_ols(df, 'log_deaths', cols2)
results['severity_controls'] = res2

# Model 3: add year trend (elapsedyrs)
cols3 = ['masfem', 'wind', 'min', 'category', 'elapsedyrs']
res3 = run_ols(df, 'log_deaths', cols3)
results['severity_plus_time'] = res3

# Model 4: add log damage (proxy for exposure/economic impact)
cols4 = ['masfem', 'wind', 'min', 'category', 'elapsedyrs', 'log_ndam15']
res4 = run_ols(df, 'log_deaths', cols4)
results['severity_time_damage'] = res4

# Model 5: interaction with category (masfem * category)
# This is similar to the original claim that femininity matters more for stronger storms

df['masfem_x_cat'] = df['masfem'] * df['category']
cols5 = ['masfem', 'category', 'masfem_x_cat', 'wind', 'min', 'elapsedyrs']
res5 = run_ols(df, 'log_deaths', cols5)
results['interaction_category'] = res5

# Alternative measure: binary gender
cols6 = ['gender_mf', 'wind', 'min', 'category', 'elapsedyrs']
res6 = run_ols(df, 'log_deaths', cols6)
results['gender_binary'] = res6

# Print concise summary

def coef_table(res, var):
    if var not in res.params:
        return None
    return {
        'coef': float(res.params[var]),
        'se': float(res.bse[var]),
        'p': float(res.pvalues[var]),
        'n': int(res.nobs)
    }

summary = {}
for name, res in results.items():
    entry = {
        'masfem': coef_table(res, 'masfem'),
        'gender_mf': coef_table(res, 'gender_mf'),
        'masfem_x_cat': coef_table(res, 'masfem_x_cat')
    }
    summary[name] = entry

print('SUMMARY')
for name, entry in summary.items():
    print(name, entry)

# Also output correlations
corr = df[['masfem', 'alldeaths', 'log_deaths', 'wind', 'min', 'category', 'log_ndam15']].corr()
print('\nCORR')
print(corr)

