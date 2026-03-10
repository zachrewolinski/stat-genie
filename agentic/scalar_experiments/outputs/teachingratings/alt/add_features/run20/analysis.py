import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

# Load data

df = pd.read_csv('teachingratings.csv')

# Keep relevant columns
cols = ['eval', 'beauty', 'age', 'gender', 'minority', 'native', 'tenure', 'division', 'credits', 'students', 'allstudents']
df = df[cols].copy()

# Drop rows with missing values in any of these columns
clean = df.dropna()

# Simple correlation
corr = clean['eval'].corr(clean['beauty'])
pearson = stats.pearsonr(clean['beauty'], clean['eval'])

# Simple OLS: eval ~ beauty
model_simple = smf.ols('eval ~ beauty', data=clean).fit(cov_type='HC3')

# Multiple OLS with controls
model_controls = smf.ols(
    'eval ~ beauty + age + C(gender) + C(minority) + C(native) + C(tenure) + C(division) + C(credits) + students + allstudents',
    data=clean
).fit(cov_type='HC3')

print('N', len(clean))
print('corr', corr)
print('pearson_r', pearson.statistic, 'p', pearson.pvalue)
print('\nSimple OLS')
print(model_simple.summary())
print('\nControls OLS')
print(model_controls.summary())
