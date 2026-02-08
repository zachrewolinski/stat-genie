import pandas as pd
import statsmodels.api as sm
import numpy as np

# Load data
_df = pd.read_csv('amtl.csv')

# Basic cleaning: drop rows with missing critical fields
cols_needed = ['num_amtl', 'sockets', 'age', 'prob_male', 'genus', 'tooth_class']
df = _df.dropna(subset=cols_needed).copy()

# Ensure valid sockets
mask = df['sockets'] > 0
if not mask.all():
    df = df.loc[mask].copy()

# Response as proportion with binomial weights
endog = df['num_amtl'] / df['sockets']

# Predictors: human indicator + age + sex prob + tooth class dummies
X = pd.DataFrame({
    'is_human': (df['genus'] == 'Homo sapiens').astype(int),
    'age': df['age'],
    'prob_male': df['prob_male'],
})

# Tooth class dummies (baseline: Anterior)
class_dummies = pd.get_dummies(df['tooth_class'], prefix='tooth', drop_first=True)
X = pd.concat([X, class_dummies], axis=1)

X = sm.add_constant(X, has_constant='add')

# Fit binomial GLM with weights as number of trials
model = sm.GLM(endog, X, family=sm.families.Binomial(), var_weights=df['sockets'])
result = model.fit()

coef = result.params.get('is_human', np.nan)
se = result.bse.get('is_human', np.nan)
z = coef / se if np.isfinite(coef) and np.isfinite(se) and se != 0 else np.nan
p = result.pvalues.get('is_human', np.nan)

# Map evidence to Likert scale [-100, 100]
# Use z-score magnitude as strength, capped at 100. z=5 -> 100, z=2 -> 40, z=1 -> 20.
if np.isfinite(z):
    strength = min(100, int(round(abs(z) * 20)))
    score = strength if coef > 0 else -strength
else:
    score = 0

# Persist conclusion
with open('conclusion.txt', 'w') as f:
    f.write(str(int(score)))

# Print summary for traceability
print(result.summary())
print('\nHuman effect coef:', coef)
print('SE:', se)
print('z:', z)
print('p:', p)
print('Score:', score)
