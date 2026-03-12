import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


df = pd.read_csv('amtl.csv')

# AMTL rate per sockets
# assume 'genus' column stores AMTL count per tooth class
# and 'age' column stores observable sockets for that class

df = df.copy()
df['amtl_rate'] = df['genus'] / df['age']

df = df[df['age'] > 0]

model = smf.ols(
    'amtl_rate ~ C(tooth_class, Treatment(reference="Homo sapiens")) + pop + stdev_age + C(sockets)',
    data=df
).fit(cov_type='HC3')

# Predicted mean rate by genus at observed covariates (marginal means)
pred = model.predict(df)
mean_by_genus = pd.DataFrame({'tooth_class': df['tooth_class'], 'pred': pred}).groupby('tooth_class')['pred'].mean()

print('Predicted mean AMTL rate by genus')
print(mean_by_genus)

# Differences vs Homo
for genus in mean_by_genus.index:
    if genus == 'Homo sapiens':
        continue
    diff = mean_by_genus['Homo sapiens'] - mean_by_genus[genus]
    print('Homo -', genus, diff)

print('\nCoefficients vs Homo (rate units):')
for k, v in model.params.items():
    if k.startswith('C(tooth_class'):
        print(k, 'coef', v, 'p', model.pvalues[k])

