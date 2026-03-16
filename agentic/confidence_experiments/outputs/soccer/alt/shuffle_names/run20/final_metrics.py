import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats


df = pd.read_csv('soccer.csv')
skin = df['rater1'].add(df['nExp']).div(2)

games = df['redCards']

col = 'yellowCards'  # red cards variable

data = pd.DataFrame({'skin': skin, 'games': games, 'y': df[col]}).dropna()

data['rate'] = data['y'] / data['games']

dark = data['skin'] > 0.5
light = data['skin'] < 0.5

mean_dark = data.loc[dark, 'rate'].mean()
mean_light = data.loc[light, 'rate'].mean()

# t-test
_, pval = stats.ttest_ind(data.loc[dark, 'rate'], data.loc[light, 'rate'], equal_var=False)

# Poisson regression with offset
X = sm.add_constant(data['skin'])
model = sm.GLM(data['y'], X, family=sm.families.Poisson(), offset=np.log(data['games']))
res = model.fit()
coef = res.params['skin']
se = res.bse['skin']

irr = np.exp(coef)
ci_low = np.exp(coef - 1.96*se)
ci_high = np.exp(coef + 1.96*se)

print('dark_rate', mean_dark, 'light_rate', mean_light, 'rate_ratio', mean_dark/mean_light, 'p_ttest', pval)
print('poisson_coef', coef, 'p', res.pvalues['skin'], 'IRR', irr, 'CI', (ci_low, ci_high))
