import pandas as pd
import statsmodels.formula.api as smf

df = pd.read_csv('teachingratings.csv')

formula = 'eval ~ beauty + age + C(gender) + C(minority) + C(native) + C(tenure) + C(division) + C(credits) + students + allstudents'

m = smf.ols(formula, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['prof']})
print(m.summary())

# alternative without both students and allstudents (use students only)
formula2 = 'eval ~ beauty + age + C(gender) + C(minority) + C(native) + C(tenure) + C(division) + C(credits) + students'
m2 = smf.ols(formula2, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['prof']})
print(m2.summary())

# alternative with allstudents only
formula3 = 'eval ~ beauty + age + C(gender) + C(minority) + C(native) + C(tenure) + C(division) + C(credits) + allstudents'
m3 = smf.ols(formula3, data=df).fit(cov_type='cluster', cov_kwds={'groups': df['prof']})
print(m3.summary())
