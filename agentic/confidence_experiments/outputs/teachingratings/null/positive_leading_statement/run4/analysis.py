import json
import pandas as pd
import statsmodels.formula.api as smf

# Load data
csv_path = 'teachingratings.csv'
df = pd.read_csv(csv_path)

# Basic cleaning: ensure categorical columns are treated as category
cat_cols = ['minority','gender','credits','division','native','tenure']
for c in cat_cols:
    if c in df.columns:
        df[c] = df[c].astype('category')

# Correlation between beauty and eval
corr = df['beauty'].corr(df['eval'])

# Simple OLS
model1 = smf.ols('eval ~ beauty', data=df).fit()

# Multivariate OLS with controls
controls = 'age + gender + minority + native + tenure + division + credits + students + allstudents'
model2 = smf.ols(f'eval ~ beauty + {controls}', data=df).fit()

# Collect key stats
results = {
    'n': int(model1.nobs),
    'corr_beauty_eval': corr,
    'model1': {
        'coef_beauty': model1.params['beauty'],
        'p_beauty': model1.pvalues['beauty'],
        'ci_beauty': model1.conf_int().loc['beauty'].tolist(),
        'r2': model1.rsquared,
    },
    'model2': {
        'coef_beauty': model2.params['beauty'],
        'p_beauty': model2.pvalues['beauty'],
        'ci_beauty': model2.conf_int().loc['beauty'].tolist(),
        'r2': model2.rsquared,
    }
}

print(json.dumps(results, indent=2))
