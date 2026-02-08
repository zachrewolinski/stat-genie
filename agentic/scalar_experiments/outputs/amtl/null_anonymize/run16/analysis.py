import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
import numpy as np

# Load data

df = pd.read_csv('amtl.csv')

# Rename columns for clarity

df = df.rename(columns={
    'feature1': 'tooth_class',
    'feature2': 'specimen_id',
    'feature3': 'missing',
    'feature4': 'observable',
    'feature5': 'age',
    'feature6': 'age_uncertainty',
    'feature7': 'sex',
    'feature8': 'genus',
    'feature9': 'region',
})

# Drop rows with missing key fields

df = df.dropna(subset=['missing', 'observable', 'age', 'sex', 'tooth_class', 'genus'])

# Ensure numeric counts

df['missing'] = df['missing'].astype(float)
df['observable'] = df['observable'].astype(float)

# Guard against zero trials

df = df[df['observable'] > 0]

# Response as proportion with weights

df['prop_missing'] = df['missing'] / df['observable']

# Create design matrix with categorical variables

formula = 'prop_missing ~ C(genus) + age + sex + C(tooth_class)'

model = sm.GLM.from_formula(
    formula,
    data=df,
    family=sm.families.Binomial(),
    freq_weights=df['observable']
)

result = model.fit()

# Extract coefficient for Homo sapiens vs baseline
# Build contrast: Homo sapiens minus mean of non-human genera

params = result.params
cov = result.cov_params()

levels = sorted(df['genus'].unique())
baseline = levels[0]

# Build contrast vector aligned to params

vec = np.zeros(len(params))
name_to_idx = {name: i for i, name in enumerate(params.index)}


def add_level(level, weight):
    if level == baseline:
        vec[name_to_idx['Intercept']] += weight
    else:
        name = f'C(genus)[T.{level}]'
        if name in name_to_idx:
            vec[name_to_idx[name]] += weight

homo_level = 'Homo sapiens'
non_human_levels = [lvl for lvl in levels if lvl != homo_level]

add_level(homo_level, 1.0)
for lvl in non_human_levels:
    add_level(lvl, -1.0 / len(non_human_levels))

estimate = float(vec @ params.values)
se = float(np.sqrt(vec @ cov.values @ vec))

z = estimate / se if se > 0 else np.nan
p = 2 * (1 - norm.cdf(abs(z))) if np.isfinite(z) else np.nan

odds_ratio = float(np.exp(estimate))

# Determine scalar based on effect size and p-value

abs_est = abs(estimate)
if abs_est < 0.1:
    base = 10
elif abs_est < 0.3:
    base = 25
elif abs_est < 0.6:
    base = 45
elif abs_est < 1.0:
    base = 65
elif abs_est < 1.5:
    base = 80
else:
    base = 90

if np.isnan(p):
    adj = -15
elif p < 0.001:
    adj = 15
elif p < 0.01:
    adj = 10
elif p < 0.05:
    adj = 5
elif p < 0.1:
    adj = 0
else:
    adj = -10

score = base + adj
score = min(100, max(0, score))

if estimate < 0:
    score = -score

score_int = int(round(score))

with open('conclusion.txt', 'w') as f:
    f.write(str(score_int))

print('baseline:', baseline)
print('levels:', levels)
print('estimate:', estimate)
print('se:', se)
print('z:', z)
print('p:', p)
print('odds_ratio:', odds_ratio)
print('score:', score_int)
