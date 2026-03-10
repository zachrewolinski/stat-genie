import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats

# Load data
path = 'teachingratings.csv'
df = pd.read_csv(path)

# Basic info
n = len(df)

# Correlation between beauty (feature6) and rating (feature7)
beauty = df['feature6']
rating = df['feature7']
pearson_r, pearson_p = stats.pearsonr(beauty, rating)

# OLS regression: rating on beauty with controls
# Define formula with categorical controls
formula = (
    'feature7 ~ feature6 + C(feature2) + feature3 + C(feature4) + C(feature5) '
    '+ C(feature8) + C(feature9) + C(feature10) + feature11 + feature12'
)
model = smf.ols(formula, data=df).fit(cov_type='HC3')

coef = model.params['feature6']
se = model.bse['feature6']
# robust p-value
pval = model.pvalues['feature6']

# Standardized effect: 1 SD change in beauty -> change in rating in SD units
# compute via standardized coefficients
beauty_sd = beauty.std(ddof=1)
rating_sd = rating.std(ddof=1)
std_beta = coef * beauty_sd / rating_sd

# Also compute simple regression without controls
model_simple = smf.ols('feature7 ~ feature6', data=df).fit(cov_type='HC3')
coef_simple = model_simple.params['feature6']
pval_simple = model_simple.pvalues['feature6']

print(f"n={n}")
print(f"Pearson r={pearson_r:.4f}, p={pearson_p:.4g}")
print(f"OLS w/ controls coef={coef:.4f}, SE={se:.4f}, p={pval:.4g}, std_beta={std_beta:.4f}")
print(f"OLS simple coef={coef_simple:.4f}, p={pval_simple:.4g}")
print(model.summary())
