import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Load and rename

df = pd.read_csv('panda_nuts.csv').rename(columns={
    'nuts_opened': 'sex',
    'sex': 'hammer_type',
    'help': 'nuts_opened_count',
    'chimpanzee': 'duration_seconds',
    'seconds': 'helped'
})

help_map = {'y': 1, 'Y': 1, 'yes': 1, 'Yes': 1, 'N': 0, 'n': 0, 'no': 0, 'No': 0}
df['helped'] = df['helped'].map(help_map)

formula = 'nuts_opened_count ~ age + C(sex) + helped'
model = smf.glm(formula=formula, data=df, family=sm.families.NegativeBinomial(), offset=np.log(df['duration_seconds']))
res = model.fit()

params = res.params
conf = res.conf_int()

# rate ratios
rr = np.exp(params)
rr_conf = np.exp(conf)

print('Rate ratios (exp(coef)) and 95% CI:')
for term in rr.index:
    print(term, rr[term], rr_conf.loc[term,0], rr_conf.loc[term,1])

print('p-values:')
print(res.pvalues)
