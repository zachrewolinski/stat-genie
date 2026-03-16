import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats


df = pd.read_csv('teachingratings.csv')

# Key variables
beauty = df['feature6']
rating = df['feature7']

# Pearson correlation
corr, corr_p = stats.pearsonr(beauty, rating)

# Simple OLS
simple_model = sm.OLS(rating, sm.add_constant(beauty)).fit()

# Multiple regression with controls (categoricals as dummies)
# Exclude feature1 (row id) and feature13 (instructor id) to avoid overfitting with many fixed effects
# Use feature2 (minority), feature4 (gender), feature5 (single credit), feature8 (upper/lower),
# feature9 (native English), feature10 (tenure track) as categorical
# feature3 (age), feature11 (students in evaluation), feature12 (students enrolled) as numeric
formula = (
    'feature7 ~ feature6 + feature3 + feature11 + feature12 '
    '+ C(feature2) + C(feature4) + C(feature5) + C(feature8) + C(feature9) + C(feature10)'
)
mult_model = smf.ols(formula, data=df).fit(cov_type='HC3')

# Extract metrics
simple_slope = simple_model.params['feature6']
simple_p = simple_model.pvalues['feature6']
simple_r2 = simple_model.rsquared

mult_slope = mult_model.params['feature6']
mult_p = mult_model.pvalues['feature6']
mult_r2 = mult_model.rsquared

beauty_sd = beauty.std(ddof=1)
rating_sd = rating.std(ddof=1)

# Effect sizes
# predicted change in rating for 1 SD increase in beauty
simple_sd_effect = simple_slope * beauty_sd
mult_sd_effect = mult_slope * beauty_sd

# Standardized beta in simple model: slope * sd_x / sd_y
simple_beta = simple_slope * beauty_sd / rating_sd
mult_beta = mult_slope * beauty_sd / rating_sd

# Print results
print('n', len(df))
print('corr', corr, 'p', corr_p)
print('simple_slope', simple_slope, 'p', simple_p, 'r2', simple_r2)
print('simple_sd_effect', simple_sd_effect, 'std_beta', simple_beta)
print('mult_slope', mult_slope, 'p', mult_p, 'r2', mult_r2)
print('mult_sd_effect', mult_sd_effect, 'std_beta', mult_beta)

# Also compute 95% CI for slope in multiple model
mult_ci = mult_model.conf_int().loc['feature6'].tolist()
print('mult_ci', mult_ci)

# and simple model CI
simple_ci = simple_model.conf_int().loc['feature6'].tolist()
print('simple_ci', simple_ci)
