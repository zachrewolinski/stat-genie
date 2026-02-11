import json
import math
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Load data
df = pd.read_csv('amtl.csv')

# Rename for clarity
df = df.rename(
    columns={
        'feature1': 'tooth_class',
        'feature3': 'missing',
        'feature4': 'total',
        'feature5': 'age',
        'feature7': 'sex',
        'feature8': 'genus',
    }
)

# Basic cleaning
df = df.dropna(subset=['tooth_class', 'missing', 'total', 'age', 'sex', 'genus'])
df = df[df['total'] > 0]
df = df[df['missing'] >= 0]
df = df[df['missing'] <= df['total']]

# Indicator for Homo sapiens vs non-human genera
df['human'] = (df['genus'] == 'Homo sapiens').astype(int)
df['prop'] = df['missing'] / df['total']

# Fit binomial regression with controls
model = smf.glm(
    formula='prop ~ human + age + sex + C(tooth_class)',
    data=df,
    family=sm.families.Binomial(),
    freq_weights=df['total'],
)
result = model.fit()

coef = float(result.params.get('human', np.nan))
se = float(result.bse.get('human', np.nan))
pval = float(result.pvalues.get('human', np.nan))

# Determine response
response = 'Yes' if (pval < 0.05 and coef > 0) else 'No'

# Map z-score to 0-100 scale using tanh for smoothness
if math.isfinite(se) and se > 0:
    z = coef / se
else:
    z = 0.0

scale = int(round(50 + 50 * math.tanh(z / 3.0)))
scale = max(0, min(100, scale))

# Write conclusion
with open('conclusion.txt', 'w') as f:
    json.dump({'response': response, 'scale': scale}, f)
