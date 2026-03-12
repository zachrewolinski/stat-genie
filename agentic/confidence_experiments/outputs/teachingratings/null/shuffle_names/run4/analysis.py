import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

# Load data
csv_path = 'teachingratings.csv'

df = pd.read_csv(csv_path)

# Basic info
print('Rows:', len(df))

# Pearson correlation
corr, pval = stats.pearsonr(df['beauty'], df['allstudents'])
print('Pearson r (beauty vs allstudents):', corr, 'p=', pval)

# Simple OLS
m1 = smf.ols('allstudents ~ beauty', data=df).fit()
print('\nSimple OLS: allstudents ~ beauty')
print(m1.summary().tables[1])

# Multiple OLS with controls (excluding likely identifiers)
# Treat categorical vars as factors
formula = (
    'allstudents ~ beauty + age + C(tenure) + C(prof) + C(native) '
    '+ C(gender) + C(credits) + rownames + minority'
)

m2 = smf.ols(formula, data=df).fit()
print('\nControlled OLS:', formula)
print(m2.summary().tables[1])

# Standardized effect for beauty in controlled model
# Standardize beauty and outcome for standardized beta
z_df = df.copy()
for col in ['beauty', 'allstudents']:
    z_df[col] = (z_df[col] - z_df[col].mean()) / z_df[col].std(ddof=0)

m2_std = smf.ols(formula.replace('allstudents', 'allstudents'), data=z_df).fit()
print('\nControlled OLS with standardized outcome/beauty')
print('Std beta (beauty):', m2_std.params['beauty'], 'p=', m2_std.pvalues['beauty'])

# Effect size: change in eval for +1 SD beauty
beauty_sd = df['beauty'].std(ddof=0)
coef_beauty = m2.params['beauty']
print('Beauty SD:', beauty_sd)
print('Controlled model: effect per +1 SD beauty:', coef_beauty * beauty_sd)
