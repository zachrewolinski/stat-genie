import pandas as pd
import statsmodels.formula.api as smf

# Load data
path = 'panda_nuts.csv'
df = pd.read_csv(path)

# Map columns to semantics based on metadata descriptions
# age in years is in column 'hammer'
# sex is in column 'nuts_opened'
# help indicator is in column 'seconds'
# nuts opened count is in column 'help'
# session duration (seconds) is in column 'chimpanzee'

# Build analysis dataset
analysis = df.copy()
analysis['age_years'] = analysis['hammer']
analysis['sex'] = analysis['nuts_opened']
analysis['helped'] = analysis['seconds'].str.upper().map({'Y': 1, 'YES': 1, 'N': 0})
analysis['nuts_opened_count'] = analysis['help']
analysis['session_seconds'] = analysis['chimpanzee']
analysis['efficiency'] = analysis['nuts_opened_count'] / analysis['session_seconds']

# Drop rows with missing values just in case
analysis = analysis.dropna(subset=['age_years','sex','helped','efficiency'])

print('Rows:', len(analysis))
print(analysis[['age_years','sex','helped','efficiency']].describe(include='all'))

# Fit OLS with categorical sex and help indicator
model = smf.ols('efficiency ~ age_years + C(sex) + helped', data=analysis).fit(cov_type='HC3')
print(model.summary())

# Also show mean efficiency by group
print('\nMean efficiency by sex:')
print(analysis.groupby('sex')['efficiency'].mean())
print('\nMean efficiency by help:')
print(analysis.groupby('helped')['efficiency'].mean())

# Correlation with age
print('\nCorrelation age vs efficiency:', analysis['age_years'].corr(analysis['efficiency']))
