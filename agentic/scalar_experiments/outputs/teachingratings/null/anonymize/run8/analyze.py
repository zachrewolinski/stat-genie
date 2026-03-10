import pandas as pd
import statsmodels.api as sm
import numpy as np

# Load data
path = '/home/chenwang/stat-genie/agentic/scalar_experiments/outputs/teachingratings/null/anonymize/run8/teachingratings.csv'
df = pd.read_csv(path)

# Variables
beauty = df['feature6']
rating = df['feature7']

# Correlation
corr = beauty.corr(rating)

# Simple OLS
X_simple = sm.add_constant(beauty)
model_simple = sm.OLS(rating, X_simple).fit()

# Multiple OLS with controls (excluding feature1 id and feature13 instructor id)
categorical = ['feature2', 'feature4', 'feature5', 'feature8', 'feature9', 'feature10']
num = ['feature3', 'feature6', 'feature11', 'feature12']
X = df[num + categorical].copy()
X = pd.get_dummies(X, columns=categorical, drop_first=True)
X = sm.add_constant(X)
model_full = sm.OLS(rating, X).fit()

# Standardized effect for simple model
beauty_std = (beauty - beauty.mean()) / beauty.std(ddof=0)
rating_std = (rating - rating.mean()) / rating.std(ddof=0)
X_std = sm.add_constant(beauty_std)
model_std = sm.OLS(rating_std, X_std).fit()

# Partial correlation controlling for other covariates
controls = df[['feature3', 'feature11', 'feature12']].copy()
controls = pd.concat([controls, pd.get_dummies(df[categorical], drop_first=True)], axis=1)
controls = sm.add_constant(controls)
resid_rating = sm.OLS(rating, controls).fit().resid
resid_beauty = sm.OLS(beauty, controls).fit().resid
partial_corr = np.corrcoef(resid_rating, resid_beauty)[0, 1]

print('n', len(df))
print('corr', corr)
print('simple coef', model_simple.params['feature6'], 'p', model_simple.pvalues['feature6'], 'r2', model_simple.rsquared)
print('full coef', model_full.params['feature6'], 'p', model_full.pvalues['feature6'], 'r2', model_full.rsquared)
print('simple standardized beta', model_std.params['feature6'])
print('partial_corr', partial_corr)
