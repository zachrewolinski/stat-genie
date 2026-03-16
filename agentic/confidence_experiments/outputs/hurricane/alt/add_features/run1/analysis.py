import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
csv_path = 'hurricane.csv'
df = pd.read_csv(csv_path)

print('shape', df.shape)
print('columns', df.columns.tolist())
print(df.head())

# Keep relevant columns if exist
cols = ['alldeaths','masfem','gender_mf','category','wind','min','ndam','ndam15','year']
existing_cols = [c for c in cols if c in df.columns]
print('existing_cols', existing_cols)

# basic stats
print(df[existing_cols].describe())

# correlation between masfem and alldeaths
if 'masfem' in df.columns and 'alldeaths' in df.columns:
    print('corr masfem vs alldeaths', df['masfem'].corr(df['alldeaths']))

# log transform deaths
if 'alldeaths' in df.columns:
    df['log_deaths'] = np.log1p(df['alldeaths'])

# regression: log_deaths ~ masfem + controls
controls = []
for c in ['category','wind','min','ndam15','year']:
    if c in df.columns:
        controls.append(c)

if 'masfem' in df.columns and 'log_deaths' in df.columns:
    formula = 'log_deaths ~ masfem'
    if controls:
        formula += ' + ' + ' + '.join(controls)
    print('formula', formula)
    model = smf.ols(formula, data=df).fit()
    print(model.summary())

# gender_mf alternative
if 'gender_mf' in df.columns and 'log_deaths' in df.columns:
    formula = 'log_deaths ~ gender_mf'
    if controls:
        formula += ' + ' + ' + '.join(controls)
    print('formula', formula)
    model2 = smf.ols(formula, data=df).fit()
    print(model2.summary())

# poisson/neg binomial with alldeaths counts
if 'alldeaths' in df.columns and 'masfem' in df.columns:
    formula = 'alldeaths ~ masfem'
    if controls:
        formula += ' + ' + ' + '.join(controls)
    print('poisson formula', formula)
    poisson = smf.glm(formula, data=df, family=sm.families.Poisson()).fit()
    print(poisson.summary())

    try:
        nb = smf.glm(formula, data=df, family=sm.families.NegativeBinomial()).fit()
        print(nb.summary())
    except Exception as e:
        print('NB error', e)
