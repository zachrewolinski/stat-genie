import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
info = json.load(open('info.json'))

df = pd.read_csv('amtl.csv')

# Rename columns for clarity
col_map = {
    'feature1': 'tooth_class',
    'feature2': 'specimen_id',
    'feature3': 'missing',
    'feature4': 'sockets',
    'feature5': 'age',
    'feature6': 'age_unc',
    'feature7': 'sex_est',
    'feature8': 'genus',
    'feature9': 'region',
}

df = df.rename(columns=col_map)

# Basic cleaning
# Ensure counts are integers and valid
for c in ['missing', 'sockets']:
    df[c] = pd.to_numeric(df[c], errors='coerce')

# Drop rows with invalid counts or missing covariates
needed = ['missing', 'sockets', 'age', 'sex_est', 'tooth_class', 'genus']

df = df.dropna(subset=needed).copy()

# Keep rows with valid binomial counts
valid = (df['sockets'] > 0) & (df['missing'] >= 0) & (df['missing'] <= df['sockets'])

df = df[valid].copy()

# Binary indicator for Homo sapiens
# Use exact label as in samples
human_label = 'Homo sapiens'

df['is_human'] = (df['genus'] == human_label).astype(int)

# Build endog as successes/failures
endog = np.column_stack([df['missing'].values, (df['sockets'] - df['missing']).values])

# Build design matrix with categorical tooth_class
# Use formula for convenience
# sex_est treated as continuous (0-1)
formula = 'missing + I(sockets - missing) ~ is_human + age + sex_est + C(tooth_class)'

# statsmodels GLM with binomial uses 2-column endog via data in formula is tricky;
# we'll build exog separately to avoid confusion.

exog = sm.add_constant(
    pd.get_dummies(df[['is_human', 'age', 'sex_est', 'tooth_class']], columns=['tooth_class'], drop_first=True)
)

model = sm.GLM(endog, exog, family=sm.families.Binomial())
res = model.fit()

# Extract coefficient for is_human
coef = res.params['is_human']
se = res.bse['is_human']

z = coef / se if se != 0 else np.sign(coef) * np.inf

# Map z to Likert scale -100..100 using tanh
score = int(np.round(100 * np.tanh(z / 2)))

# Ensure within bounds
score = max(-100, min(100, score))

# Write conclusion
with open('conclusion.txt', 'w') as f:
    f.write(str(score))

# Save a short summary for debugging (not required by instructions)
summary = {
    'coef_is_human': float(coef),
    'se_is_human': float(se),
    'z_is_human': float(z),
    'score': score,
    'n': int(len(df)),
}

with open('analysis_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
