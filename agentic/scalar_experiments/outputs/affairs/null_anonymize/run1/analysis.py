import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm


df = pd.read_csv('affairs.csv')

# Identify columns
# feature2 = affairs frequency; feature6 = children (yes/no)

# Clean / encode

df['children'] = (df['feature6'].str.lower() == 'yes').astype(int)

# outcome: any affair

df['affair_any'] = (df['feature2'] > 0).astype(int)

# group summaries

group = df.groupby('children')
mean_affairs = group['feature2'].mean()
prop_any = group['affair_any'].mean()

# t-test for mean affairs
x0 = df.loc[df['children'] == 0, 'feature2']
x1 = df.loc[df['children'] == 1, 'feature2']

t_stat, t_p = stats.ttest_ind(x1, x0, equal_var=False)

# two-proportion z-test for affair_any
n1 = df.loc[df['children'] == 1, 'affair_any'].count()
n0 = df.loc[df['children'] == 0, 'affair_any'].count()

p1 = prop_any.loc[1]
p0 = prop_any.loc[0]

p_pool = (p1*n1 + p0*n0)/(n1+n0)
se = np.sqrt(p_pool*(1-p_pool)*(1/n1 + 1/n0))
z = (p1 - p0)/se
p_z = 2*(1-stats.norm.cdf(abs(z)))

# regression with controls
# controls: gender(feature3), age(feature4), years married(feature5), religious(feature7), education(feature8), occupation(feature9), marriage rating(feature10)

# Build design matrix
X = pd.DataFrame({
    'children': df['children'],
    'age': df['feature4'],
    'years_married': df['feature5'],
    'religious': df['feature7'],
    'education': df['feature8'],
    'occupation': df['feature9'],
    'marriage_rating': df['feature10'],
    'female': (df['feature3'].str.lower() == 'female').astype(int)
})
X = sm.add_constant(X, has_constant='add')

# Logit for any affair
try:
    logit_model = sm.Logit(df['affair_any'], X).fit(disp=False)
    logit_coef = logit_model.params['children']
    logit_p = logit_model.pvalues['children']
except Exception as e:
    logit_model = None
    logit_coef = np.nan
    logit_p = np.nan

# OLS for affair frequency (simple)
ols_model = sm.OLS(df['feature2'], X).fit(cov_type='HC3')
ols_coef = ols_model.params['children']
ols_p = ols_model.pvalues['children']

print('Mean affairs by children (0=no,1=yes):')
print(mean_affairs.to_string())
print('\nProportion any affair by children:')
print(prop_any.to_string())
print(f"\nMean diff (yes-no): {mean_affairs.loc[1]-mean_affairs.loc[0]:.3f}")
print(f"t-test p-value: {t_p:.4f}")
print(f"Proportion diff (yes-no): {p1-p0:.3f}")
print(f"z-test p-value: {p_z:.4f}")
print(f"\nLogit children coef: {logit_coef:.4f}, p={logit_p:.4f}")
print(f"OLS children coef: {ols_coef:.4f}, p={ols_p:.4f}")
