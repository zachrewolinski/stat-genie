import pandas as pd
import statsmodels.formula.api as smf


df = pd.read_csv('amtl.csv')

df['is_homo'] = (df['genus'] == 'Homo sapiens').astype(int)

df['tooth_class'] = pd.Categorical(df['tooth_class'])

model = smf.ols('num_amtl ~ is_homo + age + prob_male + C(tooth_class)', data=df).fit()
print(model.summary())

