import pandas as pd
import statsmodels.formula.api as smf

df = pd.read_csv('amtl.csv')
for col in ['feature1', 'feature8']:
    df[col] = df[col].astype('category')

formula = 'feature3 ~ C(feature8, Treatment(reference="Homo sapiens")) + C(feature1) + feature5 + feature7 + feature4'
model = smf.ols(formula, data=df).fit(cov_type='HC3')
coef_table = model.summary2().tables[1]
print(coef_table.columns)
print(coef_table.head())
