import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
csv_path = 'teachingratings.csv'
df = pd.read_csv(csv_path)

# Basic info
print('Rows:', len(df))

# Correlation between beauty and eval
corr = df['beauty'].corr(df['eval'])
print('Correlation beauty-eval:', corr)

# Bivariate OLS
model_biv = smf.ols('eval ~ beauty', data=df).fit()
print('\nBivariate OLS:')
print(model_biv.summary())

# Multivariate controls
# Choose reasonable controls: age, gender, minority, native, tenure, division, credits, students (class size), allstudents
# Also include prof? but prof is id, not as fixed effects for now (94 levels). We'll skip due to complexity.
model_mult = smf.ols('eval ~ beauty + age + C(gender) + C(minority) + C(native) + C(tenure) + C(division) + C(credits) + students + allstudents', data=df).fit()
print('\nMultivariate OLS:')
print(model_mult.summary())

# Robust SE (HC3)
model_mult_robust = model_mult.get_robustcov_results(cov_type='HC3')
print('\nMultivariate OLS (HC3 robust SE):')
print(model_mult_robust.summary())

# Standardized effect: per 1 SD in beauty
beauty_sd = df['beauty'].std()
coef = model_mult.params['beauty']
print('\nBeauty SD:', beauty_sd)
print('Effect per 1 SD beauty (eval units):', coef * beauty_sd)

# Simple partial regression with only common controls? We'll rely on above.

