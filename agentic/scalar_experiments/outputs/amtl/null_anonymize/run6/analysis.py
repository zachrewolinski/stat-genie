import json
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats

df = pd.read_csv('amtl.csv')

rename_map = {
    'feature1': 'tooth_class',
    'feature2': 'specimen_id',
    'feature3': 'missing_teeth',
    'feature4': 'observable_sockets',
    'feature5': 'age',
    'feature6': 'age_uncertainty',
    'feature7': 'sex',
    'feature8': 'genus',
    'feature9': 'region',
}

df = df.rename(columns=rename_map)

needed = ['missing_teeth', 'observable_sockets', 'age', 'sex', 'tooth_class', 'genus']
df = df.dropna(subset=needed)
df = df[df['observable_sockets'] > 0]

# Human indicator
df['human'] = (df['genus'] == 'Homo sapiens').astype(int)

# Binomial response as rate with weights
df['amtl_rate'] = df['missing_teeth'] / df['observable_sockets']

model = smf.glm(
    formula='amtl_rate ~ human + age + sex + C(tooth_class)',
    data=df,
    family=sm.families.Binomial(),
    freq_weights=df['observable_sockets']
)
result = model.fit()

coef = result.params['human']
se = result.bse['human']

z = coef / se if se > 0 else 0.0
p_pos = stats.norm.sf(z)

response = 'Yes' if p_pos < 0.05 else 'No'
scale = int(round(100 * (1 - p_pos)))
scale = max(0, min(100, scale))

with open('conclusion.txt', 'w') as f:
    json.dump({'response': response, 'scale': scale}, f)

print({'response': response, 'scale': scale, 'coef': coef, 'se': se, 'z': z, 'p_pos': p_pos})
