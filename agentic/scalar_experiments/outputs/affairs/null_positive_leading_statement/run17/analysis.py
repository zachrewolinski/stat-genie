import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('affairs.csv')

# Basic comparisons
# Any affair indicator

df['any_affair'] = (df['affairs'] > 0).astype(int)

summary = df.groupby('children').agg(
    n=('affairs','size'),
    mean_affairs=('affairs','mean'),
    mean_any=('any_affair','mean')
)

# Simple difference in means
mean_affairs_yes = summary.loc['yes','mean_affairs']
mean_affairs_no = summary.loc['no','mean_affairs']
mean_any_yes = summary.loc['yes','mean_any']
mean_any_no = summary.loc['no','mean_any']

# Regression: affairs on children + controls
# children coded yes=1

df['children_yes'] = (df['children']=='yes').astype(int)

# OLS on affairs
ols = smf.ols('affairs ~ children_yes + age + yearsmarried + religiousness + education + occupation + rating + C(gender)', data=df).fit()

# Logistic on any affair
logit = smf.logit('any_affair ~ children_yes + age + yearsmarried + religiousness + education + occupation + rating + C(gender)', data=df).fit(disp=False)

# Extract coefficient and p-value for children
ols_coef = ols.params['children_yes']
ols_p = ols.pvalues['children_yes']
logit_coef = logit.params['children_yes']
logit_p = logit.pvalues['children_yes']

# Convert logit coef to odds ratio
odds_ratio = np.exp(logit_coef)

# Print summary for manual review
print('Summary by children:')
print(summary)
print('\nDifferences (yes - no):')
print('Mean affairs difference:', mean_affairs_yes - mean_affairs_no)
print('Any affair rate difference:', mean_any_yes - mean_any_no)
print('\nOLS children_yes coef:', ols_coef, 'p=', ols_p)
print('Logit children_yes coef:', logit_coef, 'p=', logit_p, 'OR=', odds_ratio)

# Also compute Cohen d for affairs

yes = df[df['children']=='yes']['affairs']
no = df[df['children']=='no']['affairs']

def cohens_d(x, y):
    nx = len(x)
    ny = len(y)
    vx = x.var(ddof=1)
    vy = y.var(ddof=1)
    pooled = ((nx-1)*vx + (ny-1)*vy)/(nx+ny-2)
    return (x.mean() - y.mean())/np.sqrt(pooled)

print('Cohen d (yes - no) affairs:', cohens_d(yes, no))
