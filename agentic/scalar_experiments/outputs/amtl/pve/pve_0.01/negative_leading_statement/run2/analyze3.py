import pandas as pd
import statsmodels.formula.api as smf

# load data

df = pd.read_csv('amtl.csv')
for col in ['genus','tooth_class','specimen']:
    df[col] = df[col].astype('category')

# Mixed effects model with random intercept by specimen
# Use human indicator

df['is_human'] = (df['genus'] == 'Homo sapiens').astype(int)

# MixedLM: need to drop rows with missing data if any

df = df.dropna(subset=['num_amtl','age','prob_male','tooth_class','specimen','is_human'])

# Fit mixed model
model = smf.mixedlm('num_amtl ~ is_human + age + prob_male + C(tooth_class)', data=df, groups=df['specimen'])

try:
    result = model.fit(method='lbfgs')
    print(result.summary())
except Exception as e:
    print('MixedLM failed:', e)

# Mixed model with genus factor
model2 = smf.mixedlm('num_amtl ~ C(genus) + age + prob_male + C(tooth_class)', data=df, groups=df['specimen'])

try:
    result2 = model2.fit(method='lbfgs')
    print(result2.summary())
except Exception as e:
    print('MixedLM genus failed:', e)

