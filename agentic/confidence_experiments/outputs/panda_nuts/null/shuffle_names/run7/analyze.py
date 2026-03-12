import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data

df = pd.read_csv('panda_nuts.csv')

print(df.head())
print(df.dtypes)
print(df.describe(include='all'))

# Map columns to actual variables based on metadata
# age variable is in column 'hammer'
# sex variable is in column 'nuts_opened'
# help indicator is in column 'seconds'
# nuts opened count is in column 'help'
# duration seconds is in column 'chimpanzee'

# rename for clarity
renamed = df.rename(columns={
    'hammer': 'age_years',
    'nuts_opened': 'sex',
    'seconds': 'helped',
    'help': 'nuts_opened',
    'chimpanzee': 'duration_sec',
    'age': 'chimp_id',
    'sex': 'hammer_type',
})

# convert types
renamed['sex'] = renamed['sex'].astype('category')
renamed['helped'] = renamed['helped'].astype('category')
renamed['hammer_type'] = renamed['hammer_type'].astype('category')

# compute efficiency rate
renamed['efficiency'] = renamed['nuts_opened'] / renamed['duration_sec']

print('Efficiency summary')
print(renamed['efficiency'].describe())

# check help counts
print('Helped value counts')
print(renamed['helped'].value_counts())
print('Sex counts')
print(renamed['sex'].value_counts())

# GLM Poisson with offset log(duration)
# Use robust SE (HC3) to account for overdispersion
# note: ensure duration >0

renamed = renamed[renamed['duration_sec'] > 0].copy()

model = smf.glm('nuts_opened ~ age_years + C(sex) + C(helped)',
                data=renamed,
                family=sm.families.Poisson(),
                offset=np.log(renamed['duration_sec']))
res = model.fit(cov_type='HC3')
print(res.summary())

# also run linear model on efficiency for comparison
lm = smf.ols('efficiency ~ age_years + C(sex) + C(helped)', data=renamed).fit(cov_type='HC3')
print(lm.summary())

# compute effect sizes for Poisson (rate ratios)
params = res.params
conf = res.conf_int()
rate_ratios = np.exp(params)
ci_low = np.exp(conf[0])
ci_high = np.exp(conf[1])
print('Rate ratios')
print(pd.DataFrame({'RR': rate_ratios, 'CI_low': ci_low, 'CI_high': ci_high}))

# compute mean efficiency by groups
print('Mean efficiency by sex')
print(renamed.groupby('sex')['efficiency'].mean())
print('Mean efficiency by helped')
print(renamed.groupby('helped')['efficiency'].mean())

# correlation age vs efficiency
print('Age-efficiency correlation', renamed['age_years'].corr(renamed['efficiency']))
