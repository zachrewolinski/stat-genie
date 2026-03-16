import pandas as pd
import numpy as np
import statsmodels.api as sm


df = pd.read_csv('soccer.csv')
skin = df['rater1'].add(df['nExp']).div(2)

games = df['redCards']
col = 'yellowCards'

data = pd.DataFrame({'skin': skin, 'games': games, 'y': df[col]}).dropna()
X = sm.add_constant(data['skin'])
model = sm.GLM(data['y'], X, family=sm.families.Poisson(), offset=np.log(data['games']))
res = model.fit()

# dispersion
pearson_chi2 = sum(res.resid_pearson**2)
df_resid = res.df_resid
print('dispersion', pearson_chi2/df_resid)
