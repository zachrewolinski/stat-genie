import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


df = pd.read_csv('amtl.csv')

df['genus'] = pd.Categorical(df['genus'], categories=['Homo sapiens','Pan','Pongo','Papio'])
df['tooth_class'] = pd.Categorical(df['tooth_class'])

model_full = smf.ols('num_amtl ~ C(genus) + age + prob_male + C(tooth_class)', data=df).fit()

# Homo vs nonhuman indicator

df['is_homo'] = (df['genus'] == 'Homo sapiens').astype(int)
model_homo = smf.ols('num_amtl ~ is_homo + age + prob_male + C(tooth_class)', data=df).fit()

sd = df['num_amtl'].std()

print('is_homo coef', model_homo.params['is_homo'])
print('is_homo p', model_homo.pvalues['is_homo'])
print('sd', sd)
print('cohen_d', model_homo.params['is_homo']/sd)

for g in ['Pan','Pongo','Papio']:
    term = f'C(genus)[T.{g}]'
    print(g, model_full.params[term], model_full.pvalues[term])

