import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pingouin as pg
from statsmodels.discrete.discrete_model import NegativeBinomial, Poisson
from scipy import stats

# Load data
df = pd.read_csv('hurricane.csv')
print('columns', df.columns.tolist())
print('rows', len(df))
print(df.head())

# Map columns per info
# feature4: femininity index (1-11). feature6: binary female indicator.
# feature8: deaths
# feature5: min pressure (lower = stronger)
# feature7: category
# feature13: max wind

# Basic summary
print('\nSummary of key vars:')
for col in ['feature4','feature6','feature8','feature5','feature7','feature13']:
    print(col, df[col].describe())

# Create log deaths
df['log_deaths'] = np.log1p(df['feature8'])

# Check correlations
print('\nCorrelations (Pearson) with deaths and log deaths:')
for col in ['feature4','feature6']:
    for target in ['feature8','log_deaths']:
        corr = df[[col,target]].corr().iloc[0,1]
        print(col, target, corr)

# Spearman correlations
print('\nCorrelations (Spearman) with deaths and log deaths:')
for col in ['feature4','feature6']:
    for target in ['feature8','log_deaths']:
        corr = df[[col,target]].corr(method='spearman').iloc[0,1]
        print(col, target, corr)

# Partial correlations controlling for intensity proxies
print('\nPartial correlations (Pearson) controlling for feature5, feature7, feature13:')
for col in ['feature4','feature6']:
    pcorr = pg.partial_corr(data=df, x=col, y='log_deaths', covar=['feature5','feature7','feature13'], method='pearson')
    print(col, pcorr[['r','p-val','n']])

# Group comparisons for binary gender indicator
print('\nGroup comparisons (feature6: 0=male, 1=female):')
g0 = df[df['feature6'] == 0]['log_deaths']
g1 = df[df['feature6'] == 1]['log_deaths']
tt = stats.ttest_ind(g1, g0, equal_var=False)
mw = stats.mannwhitneyu(df[df['feature6'] == 1]['feature8'], df[df['feature6'] == 0]['feature8'], alternative='two-sided')
print('mean log_deaths female', g1.mean(), 'male', g0.mean())
print('Welch t-test log_deaths', tt)
print('Mann-Whitney U deaths', mw)

# Regression models
# OLS on log deaths
formula_base = 'log_deaths ~ feature4'
formula_ctrl = 'log_deaths ~ feature4 + feature5 + feature7 + feature13'

ols_base = smf.ols(formula_base, data=df).fit(cov_type='HC3')
ols_ctrl = smf.ols(formula_ctrl, data=df).fit(cov_type='HC3')

print('\nOLS base:')
print(ols_base.summary())
print('\nOLS ctrl:')
print(ols_ctrl.summary())

# Alternative using binary female indicator
formula_ctrl_bin = 'log_deaths ~ feature6 + feature5 + feature7 + feature13'
ols_ctrl_bin = smf.ols(formula_ctrl_bin, data=df).fit(cov_type='HC3')
print('\nOLS ctrl binary:')
print(ols_ctrl_bin.summary())

# Negative binomial on deaths
# Add 1e-6 to avoid issues; use GLM NB
nb_formula = 'feature8 ~ feature4 + feature5 + feature7 + feature13'
nb = smf.glm(nb_formula, data=df, family=sm.families.NegativeBinomial()).fit()
print('\nNB:')
print(nb.summary())

# Binary female NB
nb_bin = smf.glm('feature8 ~ feature6 + feature5 + feature7 + feature13', data=df, family=sm.families.NegativeBinomial()).fit()
print('\nNB binary:')
print(nb_bin.summary())

# Discrete NB with estimated alpha
X = sm.add_constant(df[['feature4','feature5','feature7','feature13']])
X_bin = sm.add_constant(df[['feature6','feature5','feature7','feature13']])
y = df['feature8']

nb2 = NegativeBinomial(y, X).fit(disp=0)
print('\nNB2 (estimated alpha) feature4:')
print(nb2.summary())

nb2_bin = NegativeBinomial(y, X_bin).fit(disp=0)
print('\nNB2 (estimated alpha) feature6:')
print(nb2_bin.summary())

# Poisson for comparison
pois = Poisson(y, X).fit(disp=0)
print('\nPoisson feature4:')
print(pois.summary())

pois_bin = Poisson(y, X_bin).fit(disp=0)
print('\nPoisson feature6:')
print(pois_bin.summary())
