import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


df = pd.read_csv('teachingratings.csv')

# Basic clean: ensure types
# eval and beauty already numeric

# Correlation
corr = df['beauty'].corr(df['eval'])

# Simple linear regression
model_simple = smf.ols('eval ~ beauty', data=df).fit()

# Multiple regression with controls
# Use categorical variables as C()
# Avoid perfect multicollinearity by using one category as baseline per C()
model_controls = smf.ols(
    'eval ~ beauty + C(gender) + C(minority) + C(native) + C(tenure) + C(division) + C(credits) + age + students + allstudents',
    data=df
).fit()

# Robustness: include instructor fixed effects? Not possible because eval per course and prof id; could include C(prof) but may soak up. We'll compute too.
model_prof = smf.ols(
    'eval ~ beauty + C(prof)',
    data=df
).fit()

print('N', len(df))
print('Correlation beauty-eval:', corr)
print('\nSimple regression')
print(model_simple.summary())
print('\nControls regression')
print(model_controls.summary())
print('\nProf fixed effects')
print(model_prof.summary())
