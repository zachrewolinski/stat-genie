import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


df = pd.read_csv('teachingratings.csv')

# Basic correlation
r, p = stats.pearsonr(df['beauty'], df['allstudents'])
print('pearson_r', r, 'p', p)

# Simple OLS
model_simple = smf.ols('allstudents ~ beauty', data=df).fit()
print(model_simple.summary())

# Extended model with available covariates (excluding likely IDs)
formula = 'allstudents ~ beauty + age + rownames + minority + C(eval) + C(tenure) + C(prof) + C(native) + C(gender) + C(credits)'
model_ctrl = smf.ols(formula, data=df).fit()
print(model_ctrl.summary())

