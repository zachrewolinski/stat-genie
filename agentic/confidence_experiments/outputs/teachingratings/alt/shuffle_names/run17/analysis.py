import pandas as pd
import statsmodels.formula.api as smf

# Load data
path = 'teachingratings.csv'
df = pd.read_csv(path)

# Basic info
print('rows', len(df))
print('columns', df.columns.tolist())

# Ensure proper types for categorical vars
cat_cols = ['eval', 'tenure', 'prof', 'native', 'gender', 'credits']
for c in cat_cols:
    if c in df.columns:
        df[c] = df[c].astype('category')

# Simple correlation
corr = df[['beauty', 'allstudents']].corr().iloc[0, 1]
print('corr_beauty_allstudents', corr)

# Simple OLS: allstudents ~ beauty
m1 = smf.ols('allstudents ~ beauty', data=df).fit()
print('\nmodel_simple')
print(m1.summary())

# Multiple OLS with controls
# Use common controls based on dataset columns
formula = 'allstudents ~ beauty + age + C(tenure) + C(prof) + C(native) + C(gender) + C(credits) + students + minority + rownames'

m2 = smf.ols(formula, data=df).fit()
print('\nmodel_controls')
print(m2.summary())

# Standardized effect size for beauty in simple model
beauty_std = df['beauty'].std()
allstudents_std = df['allstudents'].std()
std_beta = m1.params['beauty'] * (beauty_std / allstudents_std)
print('std_beta_simple', std_beta)

# Standardized effect size for beauty in control model
std_beta_ctrl = m2.params['beauty'] * (beauty_std / allstudents_std)
print('std_beta_controls', std_beta_ctrl)
