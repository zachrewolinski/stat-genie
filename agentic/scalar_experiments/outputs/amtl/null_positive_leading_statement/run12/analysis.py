import pandas as pd
import numpy as np
import statsmodels.api as sm


df = pd.read_csv('amtl.csv')

# Basic cleaning
# Ensure numeric columns
for col in ['num_amtl', 'sockets', 'age', 'prob_male']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with missing critical values
needed = ['num_amtl', 'sockets', 'age', 'prob_male', 'tooth_class', 'genus']
df = df.dropna(subset=needed).copy()

# Avoid invalid rows

df = df[df['sockets'] > 0]

# Binary indicator for modern humans
# Genus values include 'Homo sapiens'
df['is_human'] = (df['genus'] == 'Homo sapiens').astype(int)

# Prepare design matrix
# tooth_class categorical, reference chosen automatically
X = pd.get_dummies(df[['is_human', 'age', 'prob_male', 'tooth_class']], drop_first=True)
X = sm.add_constant(X, has_constant='add')

# Binomial response as successes/failures
success = df['num_amtl'].astype(int)
fail = (df['sockets'] - df['num_amtl']).astype(int)
endog = np.column_stack([success, fail])

model = sm.GLM(endog, X, family=sm.families.Binomial())
result = model.fit()

beta = result.params['is_human']
se = result.bse['is_human']
z = beta / se
p = result.pvalues['is_human']

# Average marginal effect in probability space
# Predict for each row with is_human=1 and 0
X_h = X.copy()
X_nh = X.copy()
X_h['is_human'] = 1
X_nh['is_human'] = 0
pred_h = result.predict(X_h)
pred_nh = result.predict(X_nh)
mean_diff = float((pred_h - pred_nh).mean())

# Map evidence strength to Likert scalar using z-score
# tanh keeps within (-1, 1)
scalar = int(round(100 * np.tanh(z / 2)))

print('n_rows', len(df))
print('beta_is_human', beta)
print('se_is_human', se)
print('z_is_human', z)
print('p_is_human', p)
print('mean_prob_diff_human_minus_nonhuman', mean_diff)
print('scalar', scalar)

with open('conclusion.txt', 'w') as f:
    f.write(str(scalar))
