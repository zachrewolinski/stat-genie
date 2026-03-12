import pandas as pd
import statsmodels.formula.api as smf


df = pd.read_csv('teachingratings.csv')
model = smf.ols('eval ~ beauty', data=df).fit()
print(model.summary().tables[1])

print('corr', df[['eval','beauty']].corr().iloc[0,1])
