import pandas as pd
import statsmodels.formula.api as smf

# Load data
path = 'teachingratings.csv'
df = pd.read_csv(path)

# Simple bivariate model
m1 = smf.ols('eval ~ beauty', data=df).fit()

# Model with common controls (categorical variables as factors)
formula = (
    'eval ~ beauty + age + C(gender) + C(minority) + C(credits) + '
    'C(division) + C(native) + C(tenure) + students + allstudents'
)
m2 = smf.ols(formula, data=df).fit(cov_type='HC3')

# Summaries for record
print('Bivariate model: eval ~ beauty')
print(m1.summary().tables[1])
print('\nControlled model (HC3 robust SE):')
print(m2.summary().tables[1])

# Key numbers for conclusion
beauty_coef = m2.params['beauty']
beauty_p = m2.pvalues['beauty']
print(f"\nControlled model beauty coef: {beauty_coef:.4f}, p-value: {beauty_p:.4f}")
