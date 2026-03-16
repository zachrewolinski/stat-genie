import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy import stats

path = 'teachingratings.csv'

df = pd.read_csv(path)

# Identify categorical columns
cat_cols = df.select_dtypes(include='object').columns.tolist()
print('categorical', cat_cols)
for col in cat_cols:
    print(col, df[col].unique())

# Basic correlation between beauty and allstudents
r, p = stats.pearsonr(df['beauty'], df['allstudents'])
print('pearson_r', r, 'p', p)

# Simple regression
model_simple = smf.ols('allstudents ~ beauty', data=df).fit()
print(model_simple.summary())

# Build a control model with plausible covariates
# Use categorical for object columns, numeric for others (excluding outcome)
controls = []
for col in df.columns:
    if col in ['allstudents', 'beauty']:
        continue
    if df[col].dtype == 'object':
        controls.append(f'C({col})')
    else:
        controls.append(col)

formula = 'allstudents ~ beauty + ' + ' + '.join(controls)
model_controls = smf.ols(formula, data=df).fit()
print('formula', formula)
print(model_controls.summary())

# standardized effect of beauty (per 1 SD) in simple model
beauty_std = df['beauty'].std(ddof=1)
all_std = df['allstudents'].std(ddof=1)
beta_simple = model_simple.params['beauty'] * (beauty_std / all_std)
print('std_beta_simple', beta_simple)

# standardized effect of beauty in control model
beta_control = model_controls.params['beauty'] * (beauty_std / all_std)
print('std_beta_control', beta_control)

