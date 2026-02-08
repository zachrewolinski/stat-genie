import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load data
path = 'amtl.csv'
df = pd.read_csv(path)

# Keep relevant columns and drop missing
needed = ['num_amtl', 'sockets', 'age', 'prob_male', 'tooth_class', 'genus']
df = df[needed].copy()

# Drop rows with missing or invalid values
for col in ['num_amtl', 'sockets', 'age', 'prob_male']:
    df = df[pd.to_numeric(df[col], errors='coerce').notna()]

df = df.dropna(subset=['tooth_class', 'genus'])

# Ensure numeric types
for col in ['num_amtl', 'sockets', 'age', 'prob_male']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Remove rows with sockets <= 0 or num_amtl > sockets
valid = (df['sockets'] > 0) & (df['num_amtl'] >= 0) & (df['num_amtl'] <= df['sockets'])
df = df[valid].copy()

# Create human indicator
df['human'] = (df['genus'] == 'Homo sapiens').astype(int)

# Response proportion
# Use freq_weights as number of trials

df['prop'] = df['num_amtl'] / df['sockets']

# Fit binomial GLM
model = smf.glm(
    'prop ~ human + age + prob_male + C(tooth_class)',
    data=df,
    family=sm.families.Binomial(),
    freq_weights=df['sockets']
).fit()

coef = model.params['human']
se = model.bse['human']
z = coef / se
p = model.pvalues['human']
oratio = np.exp(coef)

# Average marginal effect: set human to 1 vs 0
mean_df = df.copy()
mean_df['human'] = 1
pred_h = model.predict(mean_df)
mean_df['human'] = 0
pred_nh = model.predict(mean_df)
mean_diff = (pred_h - pred_nh).mean()

print('Rows used:', len(df))
print('Human coefficient:', coef)
print('SE:', se)
print('z:', z)
print('p:', p)
print('Odds ratio:', oratio)
print('Average marginal difference (human - nonhuman):', mean_diff)
print('\nModel summary:\n', model.summary())
