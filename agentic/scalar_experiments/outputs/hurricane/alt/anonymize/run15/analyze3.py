import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm

path = 'hurricane.csv'
df = pd.read_csv(path)

mapping = {
    'feature1': 'id',
    'feature2': 'year',
    'feature3': 'name',
    'feature4': 'masfem',
    'feature5': 'min_pressure',
    'feature6': 'female',
    'feature7': 'category',
    'feature8': 'deaths',
    'feature9': 'damage_2013',
    'feature10': 'years_since',
    'feature11': 'source',
    'feature12': 'masfem_mturk',
    'feature13': 'max_wind',
    'feature14': 'damage_2015',
}

df = df.rename(columns=mapping)

# Poisson regression
pois = smf.glm('deaths ~ masfem + min_pressure + max_wind + category + year',
              data=df, family=sm.families.Poisson()).fit(cov_type='HC3')
print('poisson', float(pois.params['masfem']), float(pois.pvalues['masfem']), pois.conf_int().loc['masfem'].tolist())

# Negative binomial (discrete) with estimated alpha
try:
    nb2 = smf.negativebinomial('deaths ~ masfem + min_pressure + max_wind + category + year', data=df).fit(disp=False, cov_type='HC3')
    print('nb2', float(nb2.params['masfem']), float(nb2.pvalues['masfem']), nb2.conf_int().loc['masfem'].tolist())
    if 'alpha' in nb2.params:
        print('alpha', float(nb2.params['alpha']))
except Exception as e:
    print('nb2 error', e)

