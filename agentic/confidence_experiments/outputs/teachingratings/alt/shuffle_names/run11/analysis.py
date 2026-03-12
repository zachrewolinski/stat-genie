import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.formula.api as smf

path = 'teachingratings.csv'

df = pd.read_csv(path)
print('shape', df.shape)
print(df.head())
print(df.dtypes)
print(df.isna().sum())

# basic stats
print('beauty stats', df['beauty'].describe())
print('allstudents stats', df['allstudents'].describe())

# correlations
pearson_r, pearson_p = stats.pearsonr(df['beauty'], df['allstudents'])
spearman_r, spearman_p = stats.spearmanr(df['beauty'], df['allstudents'])
print('pearson', pearson_r, pearson_p)
print('spearman', spearman_r, spearman_p)

# simple regression
model_simple = smf.ols('allstudents ~ beauty', data=df).fit(cov_type='HC3')
print(model_simple.summary())

# multivariate regression with controls
cat_cols = [c for c in ['eval','tenure','prof','native','gender','credits'] if c in df.columns]
num_cols = [c for c in ['age','division','rownames','minority','students'] if c in df.columns]

formula = 'allstudents ~ beauty'
for c in num_cols:
    formula += f' + {c}'
for c in cat_cols:
    formula += f' + C({c})'

model_full = smf.ols(formula, data=df).fit(cov_type='HC3')
print('formula', formula)
print(model_full.summary())

# effect size: 1 SD change in beauty
beauty_sd = df['beauty'].std()
coef = model_simple.params['beauty']
coef_full = model_full.params['beauty']
print('1 SD effect simple', coef * beauty_sd)
print('1 SD effect full', coef_full * beauty_sd)

