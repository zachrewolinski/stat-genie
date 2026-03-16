import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# Load data

df = pd.read_csv('teachingratings.csv')

# rename for readability
beauty = df['feature6']
rating = df['feature7']

print('rows', len(df))
print('beauty mean', beauty.mean(), 'std', beauty.std())
print('rating mean', rating.mean(), 'std', rating.std())

# correlation
r, p = stats.pearsonr(beauty, rating)
print('pearson r', r, 'p', p)

# simple regression
X = sm.add_constant(beauty)
model = sm.OLS(rating, X).fit(cov_type='HC3')
print(model.summary())

# multivariable regression with controls
# Identify controls from metadata
# feature2 minority (yes/no), feature3 age, feature4 gender, feature5 single, feature8 upper, feature9 native, feature10 tenure, feature11 students eval, feature12 students enrolled

# create dummies for categorical
cat_cols = ['feature2', 'feature4', 'feature5', 'feature8', 'feature9', 'feature10']
X_ctrl = df[['feature6', 'feature3', 'feature11', 'feature12']].copy()

# add dummies
for col in cat_cols:
    dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
    X_ctrl = pd.concat([X_ctrl, dummies], axis=1)

X_ctrl = sm.add_constant(X_ctrl)
model_ctrl = sm.OLS(rating, X_ctrl).fit(cov_type='HC3')
print(model_ctrl.summary())

# effect size: standardized beta of beauty
# Standardize beauty and rating
z_beauty = (beauty - beauty.mean()) / beauty.std()
z_rating = (rating - rating.mean()) / rating.std()
model_std = sm.OLS(z_rating, sm.add_constant(z_beauty)).fit(cov_type='HC3')
print('std beta beauty (simple):', model_std.params[1], 'p', model_std.pvalues[1])

# partial correlation: residuals after controls
# regress beauty and rating on controls; then correlate residuals
controls = df[['feature3', 'feature11', 'feature12']].copy()
for col in cat_cols:
    controls = pd.concat([controls, pd.get_dummies(df[col], drop_first=True)], axis=1)
controls = sm.add_constant(controls)

beauty_resid = sm.OLS(beauty, controls).fit().resid
rating_resid = sm.OLS(rating, controls).fit().resid
r_partial, p_partial = stats.pearsonr(beauty_resid, rating_resid)
print('partial r', r_partial, 'p', p_partial)

