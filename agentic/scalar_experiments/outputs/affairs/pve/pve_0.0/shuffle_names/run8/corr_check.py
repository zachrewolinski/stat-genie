import pandas as pd
import statsmodels.api as sm


df = pd.read_csv('affairs.csv')

print('corr age vs occupation', df['age'].corr(df['occupation']))
X = sm.add_constant(df['occupation'])
model = sm.OLS(df['age'], X).fit()
print(model.params)
print(model.rsquared)
