import pandas as pd
import statsmodels.formula.api as smf

# Load data
DF_PATH = 'teachingratings.csv'
df = pd.read_csv(DF_PATH)

# Ensure object columns are treated as categorical for formula handling
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].astype('category')

# Descriptive stats
beauty_desc = df['beauty'].describe()
allstudents_desc = df['allstudents'].describe()

# Simple correlation
corr = df['beauty'].corr(df['allstudents'])

# Simple model: ratings ~ beauty
m_simple = smf.ols('allstudents ~ beauty', data=df).fit()

# Controlled model: include available covariates
# (Exclude `division` because it appears to be an ID / row index.)
formula = (
    'allstudents ~ beauty + age + C(eval) + C(tenure) + C(prof) + '
    'C(native) + C(gender) + C(credits) + rownames + minority + students'
)

m_ctrl = smf.ols(formula, data=df).fit()

print('=== Descriptives ===')
print('beauty:\n', beauty_desc)
print('allstudents:\n', allstudents_desc)
print('correlation(beauty, allstudents):', round(corr, 4))

print('\n=== Simple OLS: allstudents ~ beauty ===')
print('coef:', round(m_simple.params['beauty'], 4))
print('p-value:', m_simple.pvalues['beauty'])

print('\n=== Controlled OLS ===')
print('coef:', round(m_ctrl.params['beauty'], 4))
print('p-value:', m_ctrl.pvalues['beauty'])
print('95% CI:', tuple(round(x, 4) for x in m_ctrl.conf_int().loc['beauty']))
