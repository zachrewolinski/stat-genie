import pandas as pd
import statsmodels.formula.api as smf
import numpy as np

df = pd.read_csv('amtl.csv')
# create human indicator
# Ensure genus category with Homo sapiens

df['is_human'] = (df['genus'] == 'Homo sapiens').astype(int)

# OLS with human indicator
model1 = smf.ols('num_amtl ~ is_human + age + prob_male + C(tooth_class)', data=df).fit()
print(model1.summary())

# OLS with genus categories (reference: Pan maybe). We'll set reference to Pan for interpretability.
# Use Treatment with Pan.
model2 = smf.ols('num_amtl ~ C(genus, Treatment(reference="Pan")) + age + prob_male + C(tooth_class)', data=df).fit()
print(model2.summary())

# effect sizes: mean difference for human vs nonhuman controlling
coef = model1.params['is_human']
se = model1.bse['is_human']
ci_low, ci_high = model1.conf_int().loc['is_human']
print('is_human coef', coef, 'se', se, '95% CI', ci_low, ci_high, 'p', model1.pvalues['is_human'])

# predicted difference vs nonhuman at average covariates
# For linear model, coef is difference.

# Also compute adjusted means for each genus using model2
means = {}
for genus in df['genus'].unique():
    temp = df.copy()
    temp['genus'] = genus
    pred = model2.predict(temp).mean()
    means[genus] = pred
print('Adjusted mean num_amtl by genus (model2):')
for k,v in means.items():
    print(k, v)

# raw means for context
print('Raw mean num_amtl by genus:')
print(df.groupby('genus')['num_amtl'].mean())

